using UnityEngine;
using UnityEngine.UI;
using AreaTargetPlugin;

/// <summary>
/// 编辑器内演示脚本：不依赖 AR 硬件，使用模拟数据测试跟踪器生命周期。
/// 适合在 Unity Editor 中验证插件集成是否正确。
///
/// 使用方法：
/// 1. 创建空场景，添加空 GameObject 并挂载此脚本
/// 2. 在 Inspector 中设置 assetBundlePath
/// 3. 进入 Play Mode 观察 Console 输出
/// </summary>
public class AreaTargetEditorDemo : MonoBehaviour
{
    [Tooltip("资产包路径（StreamingAssets 下的相对路径或绝对路径）")]
    [SerializeField] private string assetBundlePath = "AreaTargetAssets/my_room";

    [Tooltip("模拟图像宽度")]
    [SerializeField] private int simulatedWidth = 640;

    [Tooltip("模拟图像高度")]
    [SerializeField] private int simulatedHeight = 480;

    [Tooltip("状态文本（可选）")]
    [SerializeField] private Text statusText;

    private AreaTargetTracker _tracker;
    private bool _initialized;
    private int _frameCount;

    void Start()
    {
        Debug.Log("[EditorDemo] === Area Target Plugin 编辑器演示 ===");

        _tracker = new AreaTargetTracker();

        // 1. 验证初始状态
        Debug.Log($"[EditorDemo] 初始状态: {_tracker.GetTrackingState()}");

        // 2. 尝试加载资产包
        string fullPath = System.IO.Path.IsPathRooted(assetBundlePath)
            ? assetBundlePath
            : System.IO.Path.Combine(Application.streamingAssetsPath, assetBundlePath);

        Debug.Log($"[EditorDemo] 正在加载资产包: {fullPath}");
        bool ok = _tracker.Initialize(fullPath);

        if (ok)
        {
            _initialized = true;
            Debug.Log("[EditorDemo] 资产包加载成功");
            SetStatus("已加载，每帧发送模拟数据...", Color.green);
        }
        else
        {
            Debug.LogWarning("[EditorDemo] 资产包加载失败（编辑器中正常，因为没有真实资产文件）");
            Debug.Log("[EditorDemo] 演示跟踪器生命周期（不需要真实资产包）...");
            DemoLifecycleWithoutAssets();
            SetStatus("生命周期演示完成，查看 Console", Color.yellow);
        }
    }

    void Update()
    {
        if (!_initialized) return;

        _frameCount++;

        // 构建模拟的灰度图像（随机噪声）
        byte[] fakeImage = new byte[simulatedWidth * simulatedHeight];
        for (int i = 0; i < fakeImage.Length; i++)
            fakeImage[i] = (byte)Random.Range(0, 256);

        // 构建模拟内参
        float fx = Mathf.Max(simulatedWidth, simulatedHeight) * 0.8f;
        var intrinsics = Matrix4x4.zero;
        intrinsics.m00 = fx;
        intrinsics.m11 = fx;
        intrinsics.m02 = simulatedWidth / 2f;
        intrinsics.m12 = simulatedHeight / 2f;
        intrinsics.m22 = 1f;

        var frame = new CameraFrame
        {
            ImageData = fakeImage,
            Width = simulatedWidth,
            Height = simulatedHeight,
            Intrinsics = intrinsics
        };

        TrackingResult result = _tracker.ProcessFrame(frame);

        // 每 60 帧打印一次状态
        if (_frameCount % 60 == 0)
        {
            Debug.Log($"[EditorDemo] Frame {_frameCount}: " +
                      $"State={result.State}, " +
                      $"Confidence={result.Confidence:F2}, " +
                      $"Features={result.MatchedFeatures}");

            SetStatus($"帧 {_frameCount} | {result.State} | 置信度 {result.Confidence:P0}", Color.white);
        }
    }

    /// <summary>
    /// 在没有真实资产包的情况下演示跟踪器的完整生命周期。
    /// </summary>
    private void DemoLifecycleWithoutAssets()
    {
        Debug.Log("--- 生命周期演示 ---");

        // 1. 未初始化时 ProcessFrame 应返回 LOST
        var dummyFrame = new CameraFrame
        {
            ImageData = new byte[100],
            Width = 10,
            Height = 10,
            Intrinsics = Matrix4x4.identity
        };
        var result = _tracker.ProcessFrame(dummyFrame);
        Debug.Log($"  未初始化时 ProcessFrame: State={result.State} (期望: LOST)");

        // 2. Reset 不应抛异常
        _tracker.Reset();
        Debug.Log($"  Reset 后状态: {_tracker.GetTrackingState()} (期望: INITIALIZING)");

        // 3. Dispose
        _tracker.Dispose();
        Debug.Log($"  Dispose 后状态: {_tracker.GetTrackingState()} (期望: LOST)");

        // 4. Dispose 后 Initialize 应返回 false
        bool ok = _tracker.Initialize("/fake/path");
        Debug.Log($"  Dispose 后 Initialize: {ok} (期望: False)");

        // 5. 多次 Dispose 不应抛异常
        _tracker.Dispose();
        _tracker.Dispose();
        Debug.Log("  多次 Dispose: 无异常 ✓");

        Debug.Log("--- 生命周期演示完成 ---");
    }

    private void SetStatus(string msg, Color color)
    {
        if (statusText != null)
        {
            statusText.text = msg;
            statusText.color = color;
        }
    }

    void OnDestroy()
    {
        _tracker?.Dispose();
    }
}
