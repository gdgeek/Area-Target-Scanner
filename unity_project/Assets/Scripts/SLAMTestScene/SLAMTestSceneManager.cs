using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using AreaTargetPlugin;

public class SLAMTestSceneManager : MonoBehaviour
{
    [Header("AR")]
    [SerializeField] private ARCameraManager arCameraManager;
    [SerializeField] private ARSession arSession;

    [Header("Area Target")]
    [SerializeField] private string assetSubPath = "SLAMTestAssets";

    [Header("场景引用")]
    [SerializeField] private Transform areaTargetOrigin;

    [Header("UI")]
    [SerializeField] private SLAMDebugPanel debugPanel;
    [SerializeField] private Button resetButton;

    [Header("音效")]
    [SerializeField] private SLAMAudioFeedback audioFeedback;

    private AreaTargetTracker _tracker;
    private AssetBundleLoader _loader;
    private AreaTargetPlugin.TrackingState _previousState;
    private GameObject _originCube;
    private bool _initialized;

    private float _fpsTimer;
    private int _fpsFrameCount;
    private int _totalFramesProcessed;
    private int _framesWithImage;
    private int _framesNoImage;
    private string _lastTrackingDetail = "";
    private float _lastIntrinsicsFx;
    private int _lastImageW, _lastImageH;
    private int _lastGrayscaleNonZero;

    void Start()
    {
        debugPanel?.SetStatus("脚本已加载", Color.cyan);
        if (resetButton != null) resetButton.onClick.AddListener(OnResetClicked);
        StartCoroutine(InitializeWhenReady());
    }

    private IEnumerator InitializeWhenReady()
    {
        float t = 0;
        while (ARSession.state < ARSessionState.SessionTracking)
        {
            t += Time.deltaTime;
            debugPanel?.SetStatus($"等AR {t:F0}s state={ARSession.state}", Color.yellow);
            yield return null;
        }
        debugPanel?.SetStatus($"AR就绪 state={ARSession.state}", Color.green);
        yield return new WaitForSeconds(0.3f);
        InitializeTracking();
    }

    /// <summary>
    /// 逐步初始化，每一步都显示详细结果。
    /// 不调用 _tracker.Initialize()，而是手动执行每个子步骤。
    /// </summary>
    private void InitializeTracking()
    {
        string assetPath = Path.Combine(Application.streamingAssetsPath, assetSubPath);
        var log = new List<string>();

        // Step 1: 检查路径
        log.Add($"路径: {assetPath}");
        bool pathExists = Directory.Exists(assetPath);
        log.Add($"存在: {pathExists}");

        if (pathExists)
        {
            var files = Directory.GetFiles(assetPath);
            log.Add($"文件数: {files.Length}");
            foreach (var f in files)
                log.Add($"  {Path.GetFileName(f)} ({new FileInfo(f).Length}B)");
        }
        else
        {
            log.Add("路径不存在!");
            if (Directory.Exists(Application.streamingAssetsPath))
            {
                foreach (var d in Directory.GetDirectories(Application.streamingAssetsPath))
                    log.Add($"  D:{Path.GetFileName(d)}");
                foreach (var f in Directory.GetFiles(Application.streamingAssetsPath))
                    log.Add($"  F:{Path.GetFileName(f)}");
            }
            debugPanel?.SetStatus(string.Join("\n", log), Color.red);
            return;
        }

        // Step 2: 加载资产包 (manifest.json 验证)
        _loader = new AssetBundleLoader();
        bool loadOk = _loader.Load(assetPath);
        log.Add($"资产包: {(loadOk ? "OK" : "FAIL")}");
        if (!loadOk)
        {
            log.Add($"错误: {_loader.LastError}");
            debugPanel?.SetStatus(string.Join("\n", log), Color.red);
            return;
        }
        var m = _loader.Manifest;
        log.Add($"  {m.name} v{m.version} KF:{m.keyframeCount}");
        debugPanel?.SetAssetInfo(m.name, m.version, m.keyframeCount);

        // Step 3: 加载 FeatureDB (SQLite)
        log.Add($"FeatureDB路径: {_loader.FeatureDbPath}");
        log.Add($"FeatureDB存在: {File.Exists(_loader.FeatureDbPath)}");
        if (File.Exists(_loader.FeatureDbPath))
            log.Add($"FeatureDB大小: {new FileInfo(_loader.FeatureDbPath).Length}B");

        // 先直接测试 SQLite 连接
        try
        {
            log.Add($"SQLite连接: {_loader.FeatureDbPath}");
            using (var testConn = new SQLite.SQLiteConnection(_loader.FeatureDbPath, SQLite.SQLiteOpenFlags.ReadOnly))
            {
                log.Add($"SQLite Open: OK");
                var count = testConn.ExecuteScalar<int>("SELECT COUNT(*) FROM keyframes");
                log.Add($"keyframes行数: {count}");
                count = testConn.ExecuteScalar<int>("SELECT COUNT(*) FROM vocabulary");
                log.Add($"vocabulary行数: {count}");
                count = testConn.ExecuteScalar<int>("SELECT COUNT(*) FROM features");
                log.Add($"features行数: {count}");
            }
        }
        catch (Exception ex)
        {
            log.Add($"SQLite异常: {ex.GetType().Name}");
            log.Add($"  {ex.Message}");
            if (ex.InnerException != null)
            {
                log.Add($"  Inner: {ex.InnerException.GetType().Name}");
                log.Add($"  {ex.InnerException.Message}");
            }
            debugPanel?.SetStatus(string.Join("\n", log), Color.red);
            return;
        }

        // 用 FeatureDatabaseReader 正式加载
        FeatureDatabaseReader featureDb = null;
        try
        {
            featureDb = new FeatureDatabaseReader();
            bool dbOk = featureDb.Load(_loader.FeatureDbPath);
            log.Add($"FeatureDB: {(dbOk ? "OK" : "FAIL")}");
            if (!dbOk)
            {
                debugPanel?.SetStatus(string.Join("\n", log), Color.red);
                return;
            }
            log.Add($"  KF:{featureDb.KeyframeCount} Vocab:{featureDb.Vocabulary.Count}");
        }
        catch (Exception ex)
        {
            log.Add($"FeatureDB异常: {ex.GetType().Name}");
            log.Add($"  {ex.Message}");
            if (ex.InnerException != null)
                log.Add($"  Inner: {ex.InnerException.Message}");
            debugPanel?.SetStatus(string.Join("\n", log), Color.red);
            return;
        }

        // Step 4: 测试 native library (通过反射调用 internal API)
        try
        {
            var bridgeType = typeof(AreaTargetTracker).Assembly.GetType("AreaTargetPlugin.NativeLocalizerBridge");
            if (bridgeType != null)
            {
                var createMethod = bridgeType.GetMethod("vl_create",
                    BindingFlags.Static | BindingFlags.NonPublic);
                if (createMethod != null)
                {
                    var handle = (IntPtr)createMethod.Invoke(null, null);
                    log.Add($"vl_create: {(handle != IntPtr.Zero ? $"OK({handle})" : "NULL!")}");
                    if (handle != IntPtr.Zero)
                    {
                        var destroyMethod = bridgeType.GetMethod("vl_destroy",
                            BindingFlags.Static | BindingFlags.NonPublic);
                        destroyMethod?.Invoke(null, new object[] { handle });
                    }
                }
                else
                {
                    log.Add("vl_create方法未找到");
                }
            }
            else
            {
                log.Add("NativeLocalizerBridge类型未找到!");
            }
        }
        catch (Exception ex)
        {
            var inner = ex.InnerException ?? ex;
            log.Add($"Native异常: {inner.GetType().Name}");
            log.Add($"  {inner.Message}");
        }

        // Step 5: 用 AreaTargetTracker 正式初始化
        _tracker = new AreaTargetTracker();
        bool initOk = false;
        try
        {
            initOk = _tracker.Initialize(assetPath);
            log.Add($"Tracker.Init: {(initOk ? "OK" : "FAIL")}");
        }
        catch (Exception ex)
        {
            log.Add($"Tracker.Init异常: {ex.GetType().Name}");
            log.Add($"  {ex.Message}");
        }

        if (!initOk)
        {
            debugPanel?.SetStatus(string.Join("\n", log), Color.red);
            _tracker?.Dispose();
            _tracker = null;
            return;
        }

        // Step 6: 订阅帧
        if (arCameraManager != null)
        {
            arCameraManager.frameReceived += OnCameraFrameReceived;
            log.Add("帧订阅: OK");
        }
        else
        {
            log.Add("arCameraManager=NULL!");
        }

        _initialized = true;
        _previousState = AreaTargetPlugin.TrackingState.INITIALIZING;
        log.Add("初始化完成!");
        debugPanel?.SetStatus(string.Join("\n", log), Color.green);
    }

    private void OnCameraFrameReceived(ARCameraFrameEventArgs args)
    {
        if (!_initialized || _tracker == null) return;
        _totalFramesProcessed++;

        if (!arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage cpuImage))
        {
            _framesNoImage++;
            return;
        }
        _framesWithImage++;

        var convParams = new XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, cpuImage.width, cpuImage.height),
            outputDimensions = new Vector2Int(cpuImage.width, cpuImage.height),
            outputFormat = TextureFormat.R8,
            transformation = XRCpuImage.Transformation.None
        };

        int bufSize = cpuImage.GetConvertedDataSize(convParams);
        byte[] gray = new byte[bufSize];
        unsafe
        {
            fixed (byte* ptr = gray)
                cpuImage.Convert(convParams, (IntPtr)ptr, bufSize);
        }

        _lastImageW = cpuImage.width;
        _lastImageH = cpuImage.height;
        cpuImage.Dispose();

        int nonZero = 0;
        for (int i = 0; i < Mathf.Min(gray.Length, 1000); i++)
            if (gray[i] > 0) nonZero++;
        _lastGrayscaleNonZero = nonZero;

        Matrix4x4 intrinsics = BuildIntrinsicsMatrix(_lastImageW, _lastImageH);
        _lastIntrinsicsFx = intrinsics.m00;

        var frame = new CameraFrame
        {
            ImageData = gray,
            Width = _lastImageW,
            Height = _lastImageH,
            Intrinsics = intrinsics
        };

        try
        {
            TrackingResult result = _tracker.ProcessFrame(frame);
            _lastTrackingDetail = $"S={result.State} M={result.MatchedFeatures} C={result.Confidence:F3}";
            HandleTrackingResult(result);
        }
        catch (Exception ex)
        {
            _lastTrackingDetail = $"异常:{ex.GetType().Name}:{ex.Message}";
        }
    }

    private void HandleTrackingResult(TrackingResult result)
    {
        bool toTracking = result.State == AreaTargetPlugin.TrackingState.TRACKING
            && _previousState != AreaTargetPlugin.TrackingState.TRACKING;
        bool toLost = result.State == AreaTargetPlugin.TrackingState.LOST
            && _previousState == AreaTargetPlugin.TrackingState.TRACKING;

        if (toTracking)
        {
            audioFeedback?.PlayTrackingFound();
            if (_originCube == null) _originCube = CreateOriginCube();
            _originCube.SetActive(true);
        }
        if (toLost)
        {
            audioFeedback?.PlayTrackingLost();
            if (_originCube != null) _originCube.SetActive(false);
        }

        _previousState = result.State;
    }

    private GameObject CreateOriginCube()
    {
        var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = "OriginCube";
        if (areaTargetOrigin != null) cube.transform.SetParent(areaTargetOrigin, false);
        cube.transform.localPosition = Vector3.zero;
        cube.transform.localScale = Vector3.one * 0.1f;
        return cube;
    }

    void Update()
    {
        _fpsFrameCount++;
        _fpsTimer += Time.unscaledDeltaTime;
        if (_fpsTimer >= 1f)
        {
            debugPanel?.SetFPS(_fpsFrameCount / _fpsTimer);
            _fpsTimer = 0;
            _fpsFrameCount = 0;

            if (_initialized)
            {
                debugPanel?.SetDetailedTracking(
                    $"帧:{_totalFramesProcessed} 图:{_framesWithImage} 无:{_framesNoImage}\n" +
                    $"图像:{_lastImageW}x{_lastImageH} nz:{_lastGrayscaleNonZero}/1000\n" +
                    $"fx:{_lastIntrinsicsFx:F1} {_lastTrackingDetail}\n" +
                    $"状态:{_previousState}");
            }
        }
    }

    public void OnResetClicked()
    {
        _tracker?.Reset();
        if (_originCube != null) _originCube.SetActive(false);
        debugPanel?.Clear();
        _previousState = AreaTargetPlugin.TrackingState.INITIALIZING;
    }

    void OnDestroy()
    {
        if (arCameraManager != null) arCameraManager.frameReceived -= OnCameraFrameReceived;
        _tracker?.Dispose();
        _tracker = null;
    }

    private Matrix4x4 BuildIntrinsicsMatrix(int width, int height)
    {
        if (arCameraManager != null &&
            arCameraManager.TryGetIntrinsics(out XRCameraIntrinsics arIntrinsics))
        {
            var mat = Matrix4x4.zero;
            mat.m00 = arIntrinsics.focalLength.x;
            mat.m11 = arIntrinsics.focalLength.y;
            mat.m02 = arIntrinsics.principalPoint.x;
            mat.m12 = arIntrinsics.principalPoint.y;
            mat.m22 = 1f;
            return mat;
        }
        float fx = Mathf.Max(width, height) * 0.8f;
        var fb = Matrix4x4.zero;
        fb.m00 = fx; fb.m11 = fx;
        fb.m02 = width / 2f; fb.m12 = height / 2f;
        fb.m22 = 1f;
        return fb;
    }
}
