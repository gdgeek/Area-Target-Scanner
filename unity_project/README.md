# AreaTarget Plugin — Unity 测试工程

可构建到 iOS 设备的 Unity 工程，包含两个测试场景。

## 工程结构

```
unity_project/
├── Assets/
│   ├── Scenes/
│   │   ├── TestScene.unity          # 编辑器/设备通用测试场景
│   │   └── ARTestScene.unity        # AR 真机测试场景
│   ├── Scripts/
│   │   ├── TestSceneManager.cs      # 自动化测试（Tracker、MapManager、Pipeline 等）
│   │   ├── ARTestSceneManager.cs    # AR 真机定位测试
│   │   └── SceneSelector.cs         # 场景切换
│   ├── Editor/
│   │   └── BuildiOS.cs              # iOS 构建脚本
│   └── StreamingAssets/             # 资产包放置目录
├── Packages/
│   └── manifest.json                # 引用本地 AreaTargetPlugin 包
└── ProjectSettings/                 # Unity 项目设置（iOS 平台已配置）
```

## 快速开始

### 1. 用 Unity 打开工程

用 Unity 2021.3+ 打开 `unity_project/` 目录。首次打开会自动解析 Packages。

### 2. 编辑器内测试

打开 `Assets/Scenes/TestScene.unity`，进入 Play Mode。
Console 会输出所有组件的自动化测试结果。

### 3. 构建到 iOS

方式一：菜单栏 `Build → Build iOS`
方式二：命令行
```bash
/Applications/Unity/Hub/Editor/2021.3.x/Unity.app/Contents/MacOS/Unity \
  -batchmode -quit \
  -projectPath ./unity_project \
  -executeMethod BuildiOS.Build
```

### 4. Xcode 部署

构建完成后在 `Builds/iOS/` 生成 Xcode 工程，用 Xcode 打开后签名并部署到设备。

## iOS 配置

已预配置：
- 最低 iOS 版本: 16.0
- 需要 ARKit: 是
- 相机权限描述: 已设置
- Bundle ID: com.areatarget.test
- 目标设备: iPhone + iPad

构建前需要在 Xcode 中设置你的开发者团队签名。

## 测试场景说明

### TestScene（编辑器 + 设备）
自动运行以下测试：
- Tracker 生命周期（创建、初始化、Reset、Dispose）
- MapManager（注册、查找、覆盖、注销、Clear）
- LocalizationResult（Failed 工厂方法、成功结果）
- CameraDataAdapter（内参矩阵转换）
- LocalizationPipeline（创建、参数验证）

### ARTestScene（真机 AR）
- 使用 ARKit 获取相机帧
- 实时定位并显示跟踪状态
- 定位成功后显示坐标轴指示器和测试 Cube
- 支持重置跟踪
