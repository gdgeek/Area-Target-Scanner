# Area Target Plugin — Unity 示例

本目录包含两个示例脚本，演示如何在 Unity 项目中集成 Area Target 跟踪插件。

## 文件说明

| 文件 | 用途 |
|------|------|
| `AreaTargetExampleManager.cs` | 完整 AR 示例，配合 AR Foundation 在真机上运行 |
| `AreaTargetEditorDemo.cs` | 编辑器演示，不需要 AR 硬件，验证插件集成和生命周期 |

## 快速开始

### 方式一：编辑器内测试（无需 AR 设备）

1. 在 Unity 中创建空场景
2. 创建空 GameObject，命名为 `DemoManager`
3. 将 `AreaTargetEditorDemo.cs` 挂载到该 GameObject
4. 进入 Play Mode
5. 查看 Console 窗口，会看到跟踪器生命周期的完整演示输出

> 即使没有真实的资产包文件，编辑器演示也能运行，
> 它会自动降级为生命周期验证模式。

### 方式二：真机 AR 运行

#### 前置条件

- Unity 2021.3+
- AR Foundation 5.0+（通过 Package Manager 安装）
- OpenCV for Unity（从 Asset Store 购买安装）
- 配备 LiDAR 的 iOS 设备（iPhone 12 Pro+）或支持 ARCore 的 Android 设备

#### 场景搭建步骤

1. 创建新场景，删除默认 Camera

2. 添加 AR 组件：
   - `GameObject → XR → AR Session`
   - `GameObject → XR → XR Origin (Mobile AR)`

3. 准备资产包：
   ```
   Assets/StreamingAssets/AreaTargetAssets/my_room/
   ├── manifest.json
   ├── mesh.obj
   ├── mesh.mtl
   ├── texture_atlas.png
   └── features.db
   ```
   > 这些文件由 Python 后处理管线生成

4. 创建 Area Target 内容节点：
   - 创建空 GameObject，命名为 `AreaTargetOrigin`
   - 在其下放置你想锚定到真实世界的 3D 内容（模型、UI 等）

5. 创建管理器：
   - 创建空 GameObject，命名为 `AreaTargetManager`
   - 挂载 `AreaTargetExampleManager.cs`
   - 设置 Inspector 字段：
     - `Asset Bundle Path` → `AreaTargetAssets/my_room`
     - `AR Camera Manager` → 拖入 XR Origin 下 Camera 上的 ARCameraManager
     - `Area Target Origin` → 拖入 `AreaTargetOrigin`

6. （可选）添加 UI：
   - 创建 Canvas → Text，拖入 `Status Text` 字段显示跟踪状态
   - 创建 Button，绑定 `AreaTargetManager.ResetTracking()` 方法
   - 创建 Panel 作为 Lost Indicator，拖入 `Lost Indicator UI` 字段

7. Build & Run 到真机

#### 场景层级参考

```
Scene
├── AR Session
├── XR Origin (Mobile AR)
│   └── Camera Offset
│       └── Main Camera [ARCameraManager, ARCameraBackground]
├── AreaTargetManager [AreaTargetExampleManager]
├── AreaTargetOrigin
│   ├── YourARContent        ← 你的 AR 内容
│   ├── VirtualFurniture     ← 虚拟家具等
│   └── NavigationArrows     ← 导航箭头等
└── Canvas
    ├── StatusText
    ├── ResetButton
    └── LostIndicatorPanel
```

## 跟踪状态说明

| 状态 | 含义 | 建议 UI 表现 |
|------|------|-------------|
| `INITIALIZING` | 资产包已加载，等待首次定位 | 显示"请对准扫描区域" |
| `TRACKING` | 定位成功，持续跟踪中 | 显示 AR 内容，隐藏提示 |
| `LOST` | 跟踪丢失，正在尝试重新定位 | 显示"跟踪丢失"提示 |

## API 速查

```csharp
// 创建 & 初始化
var tracker = new AreaTargetTracker();
tracker.Initialize("path/to/asset_bundle");

// 每帧处理
TrackingResult result = tracker.ProcessFrame(cameraFrame);
// result.State      → INITIALIZING / TRACKING / LOST
// result.Pose       → 4x4 位姿矩阵（相对于 Area Target 坐标系）
// result.Confidence → 置信度 [0.0, 1.0]
// result.MatchedFeatures → 匹配的特征点数

// 重置（重新定位）
tracker.Reset();

// 释放资源
tracker.Dispose();
```
