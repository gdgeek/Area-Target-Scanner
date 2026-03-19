using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using AreaTargetPlugin;
using AreaTargetPlugin.PointCloudLocalization;

/// <summary>
/// AR 真机测试场景：在 iOS 设备上使用 ARKit 进行区域定位测试。
/// 集成新的 LocalizationPipeline 端到端链路。
/// </summary>
public class ARTestSceneManager : MonoBehaviour
{
    [Header("AR 组件")]
    [SerializeField] private ARCameraManager arCameraManager;
    [SerializeField] private ARSession arSession;

    [Header("Area Target 配置")]
    [SerializeField] private string assetBundlePath = "AreaTargetAssets/test_room";
    [SerializeField] private int trackingQualityThreshold = 50;

    [Header("场景引用")]
    [SerializeField] private Transform areaTargetOrigin;
    [SerializeField] private GameObject lostIndicatorUI;
    [SerializeField] private GameObject trackingIndicatorUI;

    [Header("UI")]
    [SerializeField] private Text statusText;
    [SerializeField] private Text trackingInfoText;
    [SerializeField] private Text fpsText;
    [SerializeField] private Button resetButton;

    [Header("测试内容")]
    [SerializeField] private GameObject testCubePrefab;

    private AreaTargetTracker _tracker;
    private bool _initialized;
    private int _frameCount;
    private float _fpsTimer;
    private int _fpsFrameCount;

    void Start()
    {
        Log("=== AR 测试场景启动 ===");
        Log($"设备: {SystemInfo.deviceModel}");
        Log($"ARKit 支持: {ARSession.state}");

        if (resetButton != null)
            resetButton.onClick.AddListener(OnResetClicked);

        InitializeTracker();

        if (areaTargetOrigin != null)
            areaTargetOrigin.gameObject.SetActive(false);

        if (lostIndicatorUI != null)
            lostIndicatorUI.SetActive(true);

        if (trackingIndicatorUI != null)
            trackingIndicatorUI.SetActive(false);

        // 放置测试内容
        SpawnTestContent();
    }

    private void InitializeTracker()
    {
        _tracker = new AreaTargetTracker();

        string fullPath = System.IO.Path.IsPathRooted(assetBundlePath)
            ? assetBundlePath
            : System.IO.Path.Combine(Application.streamingAssetsPath, assetBundlePath);

        bool ok = _tracker.Initialize(fullPath);
        if (ok)
        {
            _initialized = true;
            SetStatus("初始化完成，正在定位...", Color.yellow);

            if (arCameraManager != null)
                arCameraManager.frameReceived += OnCameraFrameReceived;
        }
        else
        {
            SetStatus("资产包加载失败（编辑器模式正常）", Color.yellow);
            Log("资产包加载失败，进入模拟模式");
        }
    }

    void Update()
    {
        _fpsFrameCount++;
        _fpsTimer += Time.unscaledDeltaTime;
        if (_fpsTimer >= 1f)
        {
            float fps = _fpsFrameCount / _fpsTimer;
            _fpsTimer = 0;
            _fpsFrameCount = 0;
            if (fpsText != null)
                fpsText.text = $"FPS: {fps:F1}";
        }
    }

    private void OnCameraFrameReceived(ARCameraFrameEventArgs args)
    {
        if (!_initialized) return;

        _frameCount++;

        if (!arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage cpuImage))
            return;

        var conversionParams = new XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, cpuImage.width, cpuImage.height),
            outputDimensions = new Vector2Int(cpuImage.width, cpuImage.height),
            outputFormat = TextureFormat.R8,
            transformation = XRCpuImage.Transformation.None
        };

        int bufferSize = cpuImage.GetConvertedDataSize(conversionParams);
        byte[] grayscaleData = new byte[bufferSize];

        unsafe
        {
            fixed (byte* ptr = grayscaleData)
            {
                cpuImage.Convert(conversionParams, (IntPtr)ptr, bufferSize);
            }
        }

        int width = cpuImage.width;
        int height = cpuImage.height;
        cpuImage.Dispose();

        Matrix4x4 intrinsics = BuildIntrinsicsMatrix(width, height);

        var frame = new CameraFrame
        {
            ImageData = grayscaleData,
            Width = width,
            Height = height,
            Intrinsics = intrinsics
        };

        TrackingResult result = _tracker.ProcessFrame(frame);
        HandleTrackingResult(result);
    }

    private void HandleTrackingResult(TrackingResult result)
    {
        switch (result.State)
        {
            case AreaTargetPlugin.TrackingState.TRACKING:
                if (areaTargetOrigin != null)
                {
                    areaTargetOrigin.gameObject.SetActive(true);
                    Vector3 pos = new Vector3(result.Pose.m03, result.Pose.m13, result.Pose.m23);
                    Quaternion rot = result.Pose.rotation;
                    areaTargetOrigin.position = Vector3.Lerp(areaTargetOrigin.position, pos, 0.3f);
                    areaTargetOrigin.rotation = Quaternion.Slerp(areaTargetOrigin.rotation, rot, 0.3f);
                }

                if (lostIndicatorUI != null) lostIndicatorUI.SetActive(false);
                if (trackingIndicatorUI != null) trackingIndicatorUI.SetActive(true);

                SetStatus("跟踪中", Color.green);
                if (trackingInfoText != null)
                {
                    trackingInfoText.text = $"置信度: {result.Confidence:P0}\n" +
                                            $"特征点: {result.MatchedFeatures}\n" +
                                            $"帧: {_frameCount}";
                }
                break;

            case AreaTargetPlugin.TrackingState.LOST:
                if (lostIndicatorUI != null) lostIndicatorUI.SetActive(true);
                if (trackingIndicatorUI != null) trackingIndicatorUI.SetActive(false);
                SetStatus("跟踪丢失，请对准扫描区域", Color.red);
                break;

            case AreaTargetPlugin.TrackingState.INITIALIZING:
                SetStatus("正在初始化...", Color.yellow);
                break;
        }
    }

    private void SpawnTestContent()
    {
        if (areaTargetOrigin == null) return;

        // 在 AreaTarget 原点下创建测试几何体
        var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = "TestCube";
        cube.transform.SetParent(areaTargetOrigin);
        cube.transform.localPosition = Vector3.zero;
        cube.transform.localScale = Vector3.one * 0.3f;
        var renderer = cube.GetComponent<Renderer>();
        if (renderer != null)
            renderer.material.color = new Color(0.2f, 0.6f, 1f, 0.8f);

        // 坐标轴指示器
        CreateAxisIndicator(areaTargetOrigin);
    }

    private void CreateAxisIndicator(Transform parent)
    {
        // X 轴 - 红色
        var xAxis = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        xAxis.name = "X_Axis";
        xAxis.transform.SetParent(parent);
        xAxis.transform.localPosition = new Vector3(0.25f, 0, 0);
        xAxis.transform.localRotation = Quaternion.Euler(0, 0, 90);
        xAxis.transform.localScale = new Vector3(0.02f, 0.25f, 0.02f);
        xAxis.GetComponent<Renderer>().material.color = Color.red;

        // Y 轴 - 绿色
        var yAxis = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        yAxis.name = "Y_Axis";
        yAxis.transform.SetParent(parent);
        yAxis.transform.localPosition = new Vector3(0, 0.25f, 0);
        yAxis.transform.localScale = new Vector3(0.02f, 0.25f, 0.02f);
        yAxis.GetComponent<Renderer>().material.color = Color.green;

        // Z 轴 - 蓝色
        var zAxis = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        zAxis.name = "Z_Axis";
        zAxis.transform.SetParent(parent);
        zAxis.transform.localPosition = new Vector3(0, 0, 0.25f);
        zAxis.transform.localRotation = Quaternion.Euler(90, 0, 0);
        zAxis.transform.localScale = new Vector3(0.02f, 0.25f, 0.02f);
        zAxis.GetComponent<Renderer>().material.color = Color.blue;
    }

    private Matrix4x4 BuildIntrinsicsMatrix(int width, int height)
    {
        if (arCameraManager != null &&
            arCameraManager.TryGetIntrinsics(out XRCameraIntrinsics arIntrinsics))
        {
            var m = Matrix4x4.zero;
            m.m00 = arIntrinsics.focalLength.x;
            m.m11 = arIntrinsics.focalLength.y;
            m.m02 = arIntrinsics.principalPoint.x;
            m.m12 = arIntrinsics.principalPoint.y;
            m.m22 = 1f;
            return m;
        }

        float fx = Mathf.Max(width, height) * 0.8f;
        var fallback = Matrix4x4.zero;
        fallback.m00 = fx;
        fallback.m11 = fx;
        fallback.m02 = width / 2f;
        fallback.m12 = height / 2f;
        fallback.m22 = 1f;
        return fallback;
    }

    private void OnResetClicked()
    {
        _tracker?.Reset();
        _frameCount = 0;
        SetStatus("已重置，正在重新定位...", Color.yellow);
        if (areaTargetOrigin != null)
            areaTargetOrigin.gameObject.SetActive(false);
        if (lostIndicatorUI != null)
            lostIndicatorUI.SetActive(true);
        if (trackingIndicatorUI != null)
            trackingIndicatorUI.SetActive(false);
    }

    private void SetStatus(string msg, Color color)
    {
        if (statusText != null)
        {
            statusText.text = msg;
            statusText.color = color;
        }
    }

    private void Log(string msg)
    {
        Debug.Log($"[ARTestScene] {msg}");
    }

    void OnDestroy()
    {
        if (arCameraManager != null)
            arCameraManager.frameReceived -= OnCameraFrameReceived;
        _tracker?.Dispose();
    }
}
