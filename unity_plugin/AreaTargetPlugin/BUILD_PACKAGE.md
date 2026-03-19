# AreaTargetPlugin 打包流程

## 前置条件

- Unity 6000.x（当前使用 6000.3.11f1）
- 开发项目 `unity_project/` 已配置好，`Packages/manifest.json` 中通过 `file:` 引用了插件

## 打包内容

`.unitypackage` 包含以下内容，确保用户导入后即可编译运行：

| 路径 | 说明 |
|------|------|
| `Packages/com.areatarget.tracking/Runtime/` | 插件 C# 源码 |
| `Packages/com.areatarget.tracking/Editor/` | 编辑器扩展 |
| `Packages/com.areatarget.tracking/package.json` | UPM 包描述 |
| `Packages/com.areatarget.tracking/CHANGELOG.md` | 变更日志 |
| `Packages/com.areatarget.tracking/LICENSE.md` | 许可证 |
| `Packages/com.areatarget.tracking/README.md` | 说明文档 |
| `Assets/Plugins/macOS/*.dylib` | macOS 原生库（visual_localizer + e_sqlite3） |
| `Assets/Plugins/x86_64/*.so` | Linux 原生库 |
| `Assets/Plugins/x86_64-win/*.dll` | Windows 原生库 |
| `Assets/Plugins/Managed/*.dll` | 托管 DLL（Microsoft.Data.Sqlite + SQLitePCLRaw） |
| `Assets/link.xml` | IL2CPP 防裁剪配置 |

## 不包含的内容

- `Tests/` — 测试代码依赖 `com.unity.test-framework` 和 FsCheck，属于开发依赖，不分发给用户
- `Samples~/` — Unity 约定 `Samples~` 目录不会被导入，需通过 UPM 的 Package Manager UI 安装示例

## 打包步骤

### 方法一：Unity 菜单（推荐）

1. 用 Unity 打开 `unity_project/`
2. 菜单栏 → `Tools` → `Export AreaTargetPlugin Package`
3. 输出文件：`unity_plugin/AreaTargetPlugin/AreaTargetPlugin-{VERSION}.unitypackage`

### 方法二：命令行（CI/自动化）

```bash
UNITY_PATH="/Applications/Unity/Hub/Editor/6000.3.11f1/Unity.app/Contents/MacOS/Unity"
PROJECT_PATH="$(pwd)/unity_project"

$UNITY_PATH -batchmode -nographics \
  -projectPath "$PROJECT_PATH" \
  -executeMethod PackageExporter.Export \
  -logFile - \
  -quit
```

输出文件位于 `unity_plugin/AreaTargetPlugin/AreaTargetPlugin-{VERSION}.unitypackage`。

## 版本升级检查清单

1. 更新 `unity_plugin/AreaTargetPlugin/package.json` 中的 `version` 字段
2. 更新 `unity_plugin/AreaTargetPlugin/CHANGELOG.md`
3. 更新 `unity_project/Assets/Editor/PackageExporter.cs` 中的 `Version` 常量
4. 执行打包
5. 在空 Unity 项目中导入验证（无编译错误）
6. 提交 git 并推送

## 验证方法

```bash
# 创建空项目
$UNITY_PATH -batchmode -nographics -createProject /tmp/test_project -quit

# 导入包并编译
$UNITY_PATH -batchmode -nographics \
  -projectPath /tmp/test_project \
  -importPackage "$(pwd)/unity_plugin/AreaTargetPlugin/AreaTargetPlugin-{VERSION}.unitypackage" \
  -logFile - \
  -quit

# 检查输出中无 "error CS" 或 "Scripts have compiler errors"
# 清理
rm -rf /tmp/test_project
```
