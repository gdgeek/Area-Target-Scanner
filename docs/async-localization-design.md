# 异步定位 Pipeline 设计方案

## 背景

当前 `vl_process_frame` 在 Unity 主线程同步调用，ORB pipeline ~10ms，AKAZE fallback 额外 ~30-60ms（仅 ORB 失败时触发）。虽然同 session tracking 时 AKAZE 几乎不触发，但为了彻底消除主线程阻塞，可以将整个定位 pipeline 移到后台线程。

## AKAZE 开销分析

### 离线（一次性）
- `build_feature_database()` 阶段提取 AKAZE 描述子，写入 `akaze_features` 表
- AKAZE 提取比 ORB 慢 3-5 倍，但只在建库时跑一次
- 存储在 features.db 中，运行时通过 `vl_add_keyframe_akaze` 加载

### 运行期
- ORB 成功时：AKAZE 完全不参与，零开销
- ORB 失败时触发 fallback：
  - `akaze_->detectAndCompute()`: ~20-40ms (ARM64)
  - BFMatcher knnMatch × N 候选 KF: ~5-15ms
  - PnP RANSAC + refinement: ~1-2ms
  - 单次 fallback 总计: ~30-60ms

### 触发频率
- 同 session tracking: ORB 成功率 > 95%，AKAZE 几乎不触发
- 跨 session 重定位: ORB 失败率 ~60%，这些帧触发 AKAZE
- 跨 session 定位成功后: 后续帧 ORB 靠 nearby KF 搜索成功，AKAZE 又不触发

## 方案：后台线程定位

### 架构

```
Unity 主线程（每帧 Update）:
  1. 拷贝当前相机灰度图 + intrinsics + AR camera pose
  2. 提交给后台线程（替换待处理数据，丢弃旧帧）
  3. 读取后台线程的最新结果（如果有）
  4. 用结果更新 tracking 状态和 pose

后台线程（独立循环）:
  1. 等待新帧数据
  2. vl_process_frame（ORB → AKAZE fallback → 一致性过滤 → AT）
  3. 将结果写入共享变量
  4. 回到 1
```

### C# 端伪代码

```csharp
class AsyncLocalizationRunner : IDisposable
{
    // 共享数据（主线程写，后台线程读）
    private byte[] _pendingImage;
    private float _pendingFx, _pendingFy, _pendingCx, _pendingCy;
    private float[] _pendingPose;
    private bool _hasPendingFrame;
    private readonly object _inputLock = new object();

    // 共享结果（后台线程写，主线程读）
    private VLResult _latestResult;
    private bool _hasNewResult;
    private readonly object _outputLock = new object();

    private Thread _workerThread;
    private volatile bool _running;

    public void SubmitFrame(byte[] grayImage, float fx, float fy, 
                            float cx, float cy, float[] arPose)
    {
        lock (_inputLock)
        {
            // 深拷贝图像（主线程的 buffer 下一帧会被覆盖）
            if (_pendingImage == null || _pendingImage.Length != grayImage.Length)
                _pendingImage = new byte[grayImage.Length];
            Buffer.BlockCopy(grayImage, 0, _pendingImage, 0, grayImage.Length);
            
            _pendingFx = fx; _pendingFy = fy;
            _pendingCx = cx; _pendingCy = cy;
            
            if (_pendingPose == null) _pendingPose = new float[16];
            Array.Copy(arPose, _pendingPose, 16);
            
            _hasPendingFrame = true;
            Monitor.Pulse(_inputLock); // 唤醒后台线程
        }
    }

    public bool TryGetResult(out VLResult result)
    {
        lock (_outputLock)
        {
            if (_hasNewResult)
            {
                result = _latestResult;
                _hasNewResult = false;
                return true;
            }
            result = default;
            return false;
        }
    }

    private void WorkerLoop()
    {
        while (_running)
        {
            byte[] image; float fx, fy, cx, cy; float[] pose;
            
            lock (_inputLock)
            {
                while (!_hasPendingFrame && _running)
                    Monitor.Wait(_inputLock);
                if (!_running) break;
                
                // 取走最新帧（跳过排队中的旧帧）
                image = _pendingImage;
                fx = _pendingFx; fy = _pendingFy;
                cx = _pendingCx; cy = _pendingCy;
                pose = _pendingPose;
                _hasPendingFrame = false;
            }

            // 后台执行完整定位 pipeline
            var result = NativeBridge.vl_process_frame(
                _handle, image, width, height, fx, fy, cx, cy, 1, pose);

            lock (_outputLock)
            {
                _latestResult = result;
                _hasNewResult = true;
            }
        }
    }
}
```

### 主线程调用

```csharp
void Update()
{
    // 1. 提交当前帧
    _asyncRunner.SubmitFrame(currentGrayImage, fx, fy, cx, cy, arCameraPose);
    
    // 2. 读取上一次结果
    if (_asyncRunner.TryGetResult(out var result))
    {
        if (result.state == 1) // TRACKING
        {
            UpdatePose(result.pose);
            _trackingState = TrackingState.Tracking;
        }
        else
        {
            _trackingState = TrackingState.Lost;
        }
    }
}
```

## 注意事项

### 图像数据
- 必须深拷贝，主线程的 camera buffer 下一帧会被覆盖
- 考虑用 double buffer 避免每帧 alloc

### 帧跳过
- 后台线程只处理最新帧，跳过排队中的旧帧
- 避免定位结果积压导致延迟越来越大

### 线程安全
- `vl_process_frame` 内部只读 keyframe 数据，线程安全
- `vl_add_keyframe` / `vl_add_keyframe_akaze` / `vl_build_index` 在初始化阶段调用，与 processFrame 不并发
- `vl_set_alignment_transform` 可能运行时调用，需要加锁或用 atomic flag

### 结果延迟
- 定位结果延迟 1-2 帧（~16-33ms @60fps）
- 对 AR 体验影响很小，ARKit/ARCore 本身的 tracking 也有类似延迟
- 可以用 Kalman filter 或插值平滑 pose 过渡

### 不需要 ORB/AKAZE 分线程
- ORB 和 AKAZE 是串行关系（ORB 失败才跑 AKAZE）
- 放同一个后台线程即可
- 分两个线程增加同步复杂度，收益不大

## 实施建议

1. 先用当前同步方案验证 AKAZE 精度效果
2. 确认有用后，在 C# 端做异步改造（C++ native 端不需要改）
3. 异步改造主要改 `LocalizationPipeline.cs` 或新建 `AsyncLocalizationRunner.cs`
4. 如果需要进一步优化，可以考虑限制 AKAZE fallback 频率（每 N 帧触发一次）
