# iOS 实机测试指引

## 前置条件

- macOS + Xcode 已安装
- Unity 6000.3.11f1（已安装在 `/Applications/Unity/Hub/Editor/6000.3.11f1/`）
- iPhone/iPad 通过 USB 连接，iOS 16.0+
- Apple 开发者账号已在 Xcode 中登录

## 当前状态检查清单

| 项目 | 状态 |
|------|------|
| SLAMTestAssets（features.db, manifest.json, optimized.glb） | ✅ 已就绪 |
| ARTestSceneManager 默认路径 → SLAMTestAssets | ✅ 已修正 |
| ARTestSceneManager 资产预检查 | ✅ 已添加 |
| ARTestSceneManager 调试 UI（Quality/Mode/AKAZE/一致性） | ✅ 已添加 |
| InternalsVisibleTo("Assembly-CSharp") | ✅ 已添加 |
| BuildiOS.cs 场景列表含 ARTestScene | ✅ 已确认 |
| iOSPostProcess.cs（OpenCV/Bitcode/系统框架） | ✅ 已确认 |
| build_ios.sh 11 个符号验证 | ✅ 已完善 |
| **iOS 静态库 libvisual_localizer.a** | ⚠️ **需要重编** |

> 当前 `libvisual_localizer.a` 是 3 月 24 日的旧版本，缺少 `vl_add_keyframe_akaze` 和 `vl_set_alignment_transform`。

---

## 步骤 1：重编 iOS 静态库

```bash
cd native_visual_localizer
bash build_ios.sh
```

预期输出：
- 自动下载 OpenCV iOS framework（首次约 200MB，已有则跳过）
- 编译 arm64 静态库
- 验证全部 11 个导出符号（无 WARNING）
- 备份旧库为 `.bak`，复制新库到 `unity_project/Assets/Plugins/iOS/`

验证：
```bash
nm unity_project/Assets/Plugins/iOS/libvisual_localizer.a | grep " T _vl_"
```
应看到 11 个 `_vl_` 开头的符号。

---

## 步骤 2：查找 iOS 设备

```bash
xcrun xctrace list devices
```

记下你的设备 UDID（括号中的十六进制字符串），后续步骤用 `<DEVICE_UDID>` 代替。

---

## 步骤 3：Unity 导出 Xcode 项目

确保没有其他 Unity Editor 实例打开 `unity_project`，然后：

```bash
/Applications/Unity/Hub/Editor/6000.3.11f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -quit -nographics \
  -projectPath ./unity_project \
  -executeMethod BuildiOS.Build \
  -logFile /tmp/unity_ios_build.log
```

耗时约 2-5 分钟。查看进度：
```bash
tail -f /tmp/unity_ios_build.log
```

成功标志：日志末尾出现 `Exiting batchmode successfully now!`

> 如果想用 Development 模式（支持调试），把 `BuildiOS.Build` 改为 `BuildiOS.BuildDevelopment`。

---

## 步骤 4：Xcode 编译并安装

```bash
xcodebuild \
  -project unity_project/Builds/iOS/Unity-iPhone.xcodeproj \
  -scheme Unity-iPhone \
  -destination "platform=iOS,id=<DEVICE_UDID>" \
  -configuration Debug \
  -allowProvisioningUpdates \
  build 2>&1 | tee /tmp/xcode_build.log
```

耗时约 2-5 分钟。成功标志：`** BUILD SUCCEEDED **`

如果签名失败，用 Xcode 打开项目手动设置一次 Team：
```bash
open unity_project/Builds/iOS/Unity-iPhone.xcodeproj
```
在 Signing & Capabilities 中选择你的开发者 Team，然后重新执行上面的命令。

---

## 步骤 5：安装到设备

```bash
APP_PATH=$(find ~/Library/Developer/Xcode/DerivedData \
  -name "AreaTargetTest.app" \
  -path "*/Debug-iphoneos/*" | head -1)

xcrun devicectl device install app \
  --device <DEVICE_UDID> "$APP_PATH"
```

---

## 步骤 6：启动 App

```bash
xcrun devicectl device process launch \
  --device <DEVICE_UDID> com.areatarget.test
```

---

## 实机测试时观察什么

App 启动后会进入 ARTestScene，屏幕上会显示：

1. **状态栏**：初始化 → 正在定位 → 跟踪中 / 跟踪丢失
2. **调试面板**（trackingInfoText）：
   - 置信度：匹配置信度百分比
   - 特征点：当前帧匹配的特征点数
   - 帧：已处理帧数
   - 模式：Raw（初始）→ Aligned（AT 计算完成后）
   - 质量：NONE → RECOGNIZED → LOCALIZED
   - AKAZE：触发 / 未触发（ORB 失败时才触发）
   - 一致性：通过 / 拒绝
3. **质量指示器**（qualityText）：
   - 红色 = NONE（未识别）
   - 黄色 = RECOGNIZED（Raw 模式识别到）
   - 绿色 = LOCALIZED（Aligned 模式高精度定位）
4. **3D 内容**：蓝色立方体 + RGB 坐标轴，跟踪成功时会叠加在真实场景上

### 正常流程预期

1. 启动 → "初始化完成，正在定位..."
2. 对准扫描过的区域 → 质量从 NONE 变为 RECOGNIZED（黄色）
3. 持续跟踪 10+ 帧 → 质量变为 LOCALIZED（绿色），模式从 Raw 变为 Aligned
4. 3D 立方体稳定叠加在场景中

### 异常情况

| 现象 | 可能原因 |
|------|----------|
| "资产目录不存在: SLAMTestAssets" | StreamingAssets 中缺少 SLAMTestAssets 目录 |
| "缺少 features.db" | features.db 文件未包含在构建中 |
| 一直 NONE 不变 | 没有对准扫描过的区域，或 features.db 数据不匹配 |
| AKAZE 频繁触发 | ORB 匹配困难（光照变化大或视角差异大） |
| 一致性频繁拒绝 | 定位结果不稳定，可能需要更多特征点 |

---

## 查看设备日志

```bash
# 实时查看 App 日志（过滤 ARTestScene 标签）
xcrun devicectl device info log --device <DEVICE_UDID> 2>&1 | grep "ARTestScene"
```

---

## 快速重建流程（代码修改后）

如果只改了 C# 代码（不涉及 native 库），跳过步骤 1，从步骤 3 开始。
如果改了 C++ native 代码，从步骤 1 开始。
