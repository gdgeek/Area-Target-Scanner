# Changelog

## [1.2.0] - 2026-03-19

### Changed
- 后端移除 v1 旧管线 (pipeline.py)，统一使用 OptimizedPipeline (v2)
- 特征提取逻辑独立为 feature_extraction.py 模块
- BoW 向量测试更新为 TF-IDF + L2 归一化断言

### Added
- GitHub Actions CI/CD (Python 3.10-3.12 测试 + ruff lint)

## [1.1.0] - 2026-03-18

### Changed
- 视觉定位引擎从 OpenCvSharp 迁移到原生 C++ 库 (libvisual_localizer)
- 移除 OpenCvSharp4 依赖，减少包体积约 40MB
- 所有平台（Editor/iOS/Android/Standalone）使用统一的原生库接口

### Added
- `NativeLocalizerBridge.cs` — P/Invoke 桥接层
- 原生库支持 macOS (.dylib)、Windows (.dll)、Linux (.so)、iOS (静态链接)
- `NativeLocalizerBridgeTests.cs` — 原生桥接层单元测试（句柄生命周期、NULL 安全、结构体编组）
- `PerformanceBenchmarkTests.cs` — 性能基准测试（帧处理延迟、吞吐量、内存稳定性）

### Fixed
- iOS 平台使用 `__Internal` 静态链接，避免动态库加载问题

## [1.0.0] - 2026-03-17

### Added
- 初始版本
- AreaTargetTracker 核心跟踪器
- VisualLocalizationEngine（ORB 特征提取 + BoW 检索 + PnP 定位）
- KalmanPoseFilter 姿态平滑
- AssetBundleLoader 资产包加载
- FeatureDatabaseReader SQLite 特征数据库读取
- LocalizationPipeline 端到端定位管线
- AR Foundation 平台支持
- 完整单元测试和属性测试套件
