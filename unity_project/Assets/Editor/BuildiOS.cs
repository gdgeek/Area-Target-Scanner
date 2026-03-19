using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

/// <summary>
/// iOS 构建脚本：支持命令行和菜单栏触发构建。
/// 命令行用法: Unity -batchmode -executeMethod BuildiOS.Build -projectPath ./unity_project
/// </summary>
public class BuildiOS
{
    [MenuItem("Build/Build iOS")]
    public static void Build()
    {
        var scenes = new[]
        {
            "Assets/Scenes/TestScene.unity",
            "Assets/Scenes/ARTestScene.unity"
        };

        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = "Builds/iOS",
            target = BuildTarget.iOS,
            options = BuildOptions.None
        };

        // 确保 iOS 平台设置
        PlayerSettings.iOS.targetDevice = iOSTargetDevice.iPhoneAndiPad;
        PlayerSettings.iOS.targetOSVersionString = "16.0";
        PlayerSettings.iOS.cameraUsageDescription = "Required for AR area target tracking";
        PlayerSettings.SetApplicationIdentifier(BuildTargetGroup.iOS, "com.areatarget.test");

        BuildReport report = BuildPipeline.BuildPlayer(options);
        BuildSummary summary = report.summary;

        if (summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"[BuildiOS] 构建成功: {summary.totalSize / (1024 * 1024):F1} MB, 耗时 {summary.totalTime.TotalSeconds:F1}s");
        }
        else
        {
            Debug.LogError($"[BuildiOS] 构建失败: {summary.result}");
            foreach (var step in report.steps)
            {
                foreach (var msg in step.messages)
                {
                    if (msg.type == LogType.Error || msg.type == LogType.Warning)
                        Debug.LogError($"  {msg.content}");
                }
            }
        }
    }

    [MenuItem("Build/Build iOS (Development)")]
    public static void BuildDevelopment()
    {
        var scenes = new[]
        {
            "Assets/Scenes/TestScene.unity",
            "Assets/Scenes/ARTestScene.unity"
        };

        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = "Builds/iOS_Dev",
            target = BuildTarget.iOS,
            options = BuildOptions.Development | BuildOptions.AllowDebugging
        };

        PlayerSettings.iOS.targetDevice = iOSTargetDevice.iPhoneAndiPad;
        PlayerSettings.iOS.targetOSVersionString = "16.0";
        PlayerSettings.iOS.cameraUsageDescription = "Required for AR area target tracking";

        BuildReport report = BuildPipeline.BuildPlayer(options);
        if (report.summary.result == BuildResult.Succeeded)
            Debug.Log("[BuildiOS] Development 构建成功");
        else
            Debug.LogError("[BuildiOS] Development 构建失败");
    }
}
