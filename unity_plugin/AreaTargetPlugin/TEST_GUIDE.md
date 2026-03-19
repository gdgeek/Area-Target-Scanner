# Area Target Plugin — 测试指南

## 测试概览

| 类别 | 测试文件 | 测试数量 | 覆盖范围 |
|------|---------|---------|---------|
| 跟踪状态 | TrackingStateTests.cs | 8 | 生命周期、初始状态、Reset、Dispose |
| 初始化 | AreaTargetTrackerInitTests.cs | 6 | 资产包加载、路径验证 |
| 资产加载 | AssetBundleLoaderTests.cs | 5 | manifest 解析、文件校验 |
| 资产属性 | AssetBundleLoaderPropertyTests.cs | 4 | 属性测试、边界条件 |
| 视觉定位 | VisualLocalizationTests.cs | 8 | PnP 精度、LOST 状态、置信度 |
| PnP 属性 | PnPPropertyTests.cs | 30 | 姿态矩阵有效性（行列式、正交性） |
| Kalman 滤波 | KalmanPoseFilterTests.cs | 6 | 平滑效果、收敛性 |
| Kalman 处理器 | KalmanDataProcessorPropertyTests.cs | 4 | 数据处理链 |
| 定位集成 | LocalizerIntegrationTests.cs | 6 | 端到端管线 |
| 定位属性 | LocalizerPropertyTests.cs | 5 | 属性测试 |
| 相机适配 | CameraDataAdapterPropertyTests.cs | 3 | 内参矩阵转换 |
| 地图管理 | MapManagerPropertyTests.cs | 5 | 注册、查找、覆盖、清除 |
| 平台支持 | PlatformSupportPropertyTests.cs | 4 | 质量阈值逻辑 |
| XR 空间 | XRSpacePropertyTests.cs | 6 | Ignore 标志、处理器链 |
| 场景更新 | SceneUpdaterPropertyTests.cs | 3 | Ignore 一致性 |
| 原生桥接 | NativeLocalizerBridgeTests.cs | 15 | P/Invoke、NULL 安全、压力测试 |
| 性能基准 | PerformanceBenchmarkTests.cs | 7 | 延迟、吞吐量、内存 |

## 运行方式

### 方式一：Unity Test Runner（推荐）

1. 打开 Unity 编辑器
2. 菜单 `Window → General → Test Runner`
3. 选择 `EditMode` 标签页
4. 点击 `Run All` 运行全部测试

### 方式二：命令行

```bash
# macOS / Linux
/Applications/Unity/Hub/Editor/6000.0.x/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -runTests \
  -projectPath ./unity_project \
  -testPlatform EditMode \
  -testResults ./test_results.xml

# 查看结果
cat test_results.xml
```

### 方式三：iOS 模拟器场景测试

```bash
# 1. 编译 Xcode 工程
xcodebuild build -project ios_scanner/AreaTargetScanner.xcodeproj \
  -scheme AreaTargetScanner \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -configuration Debug \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO

# 2. 运行单元测试
xcodebuild test -project ios_scanner/AreaTargetScanner.xcodeproj \
  -scheme AreaTargetScanner \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -configuration Debug \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO

# 3. 安装到模拟器并启动
xcrun simctl install booted <path-to-app>
xcrun simctl launch booted com.areatarget.scanner
```

### 方式四：Python 后端测试

```bash
# 运行全部 Python 测试（180 个）
python3 -m pytest tests/ -v

# 仅运行原生库集成测试
python3 -m pytest tests/test_native_localizer.py -v
```

## 测试场景详解

### 1. 编辑器自动化测试场景 (TestScene)

打开 `unity_project/Assets/Scenes/TestScene.unity`，进入 Play Mode。

自动运行以下测试组：

| 测试组 | 验证内容 |
|--------|---------|
| Tracker 生命周期 | 创建 → 初始化 → ProcessFrame → Reset → Dispose |
| MapManager | 注册 → 查找 → 覆盖 → 注销 → Clear |
| LocalizationResult | Failed 工厂方法、成功结果字段 |
| CameraDataAdapter | 内参矩阵 Vector4 → Matrix4x4 转换 |
| SceneUpdater | 创建、参数验证 |
| LocalizationPipeline | 创建、依赖注入验证 |

Console 输出格式：
```
=== AreaTarget Plugin 测试场景 ===
平台: WindowsEditor
--- 测试: Tracker 生命周期 ---
✓ 创建 Tracker
✓ 初始状态为 INITIALIZING
...
测试完成: 20/20 通过
```

### 2. AR 真机测试场景 (ARTestScene)

打开 `unity_project/Assets/Scenes/ARTestScene.unity`，构建到 iOS 设备。

测试流程：
1. 启动后自动初始化 ARKit
2. 加载资产包（StreamingAssets 下）
3. 实时获取相机帧并定位
4. 定位成功后显示坐标轴和测试 Cube
5. 屏幕显示：FPS、跟踪状态、匹配特征数、置信度
6. 点击 Reset 按钮重置跟踪

验证要点：
- [ ] 启动无崩溃
- [ ] FPS ≥ 30
- [ ] 定位成功后 Cube 位置稳定
- [ ] 跟踪丢失后能自动恢复
- [ ] Reset 后重新定位正常

### 3. 原生库集成测试

通过 Python ctypes 直接调用 `libvisual_localizer.dylib`：

```bash
python3 -m pytest tests/test_native_localizer.py -v
```

覆盖 18 个测试用例：
- 句柄生命周期（创建/销毁/多实例）
- NULL 句柄安全性（所有 API 函数）
- LOST 状态一致性（空图像、无特征图像）
- 数据加载（词汇表、关键帧、索引构建）
- VLResult 结构体布局（大小、字段偏移）
- 端到端（合成 ORB 特征 → 加载关键帧 → 处理帧）

### 4. iOS Scanner 测试

```bash
xcodebuild test -project ios_scanner/AreaTargetScanner.xcodeproj \
  -scheme AreaTargetScanner \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

覆盖 30 个测试用例：
- TextureMappingPropertyTests（17 个）：UV 坐标、顶点守恒、纹理导出
- ScanDataExporterTests（13 个）：PLY 格式、位姿 JSON、图像导出

## 性能基准

| 指标 | 目标 | 测试方法 |
|------|------|---------|
| ProcessFrame 延迟 | < 5ms | 100 次迭代取平均 |
| Kalman 更新延迟 | < 1ms | 1000 次迭代取平均 |
| 吞吐量 | ≥ 30 FPS | 1 秒内最大帧数 |
| 内存增长 | < 10MB/1000帧 | GC.GetTotalMemory 对比 |

## 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| DllNotFoundException | 原生库未部署 | 将 dylib/dll/so 放入 Plugins 目录 |
| EntryPointNotFoundException | 库版本不匹配 | 重新编译原生库 |
| 测试超时 | 模拟器未启动 | `xcrun simctl boot <device-id>` |
| LOST 状态不变 | 无资产包数据 | 确认 features.db 已加载 |
