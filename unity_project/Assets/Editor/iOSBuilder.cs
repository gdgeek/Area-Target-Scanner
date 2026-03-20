using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using System.IO;

/// <summary>
/// Headless iOS build script for Unity batch mode.
/// Usage: Unity -batchmode -nographics -projectPath unity_project -executeMethod iOSBuilder.Build -quit
/// Optional args:
///   -simulator   Build for iOS Simulator (ARM64)
///   -outputPath  Custom output directory (default: ../build/xcode)
/// </summary>
public static class iOSBuilder
{
    public static void Build()
    {
        var args = System.Environment.GetCommandLineArgs();
        bool simulator = false;
        string outputPath = Path.GetFullPath(
            Path.Combine(Application.dataPath, "../../build/xcode"));

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-simulator") simulator = true;
            if (args[i] == "-outputPath" && i + 1 < args.Length)
                outputPath = args[i + 1];
        }

        // Set simulator SDK if requested
        if (simulator)
        {
            PlayerSettings.iOS.sdkVersion = iOSSdkVersion.SimulatorSDK;
            Debug.Log("[iOSBuilder] Target: iOS Simulator");
        }
        else
        {
            PlayerSettings.iOS.sdkVersion = iOSSdkVersion.DeviceSDK;
            Debug.Log("[iOSBuilder] Target: iOS Device");
        }

        // Collect enabled scenes from build settings
        var scenes = new System.Collections.Generic.List<string>();
        foreach (var scene in EditorBuildSettings.scenes)
        {
            if (scene.enabled)
                scenes.Add(scene.path);
        }

        if (scenes.Count == 0)
        {
            Debug.LogError("[iOSBuilder] No scenes in build settings.");
            EditorApplication.Exit(1);
            return;
        }

        Debug.Log($"[iOSBuilder] Building {scenes.Count} scenes to {outputPath}");

        var options = new BuildPlayerOptions
        {
            scenes = scenes.ToArray(),
            locationPathName = outputPath,
            target = BuildTarget.iOS,
            options = BuildOptions.None
        };

        var report = BuildPipeline.BuildPlayer(options);

        if (report.summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"[iOSBuilder] Build succeeded: {report.summary.totalSize} bytes");
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
        else
        {
            Debug.LogError($"[iOSBuilder] Build failed: {report.summary.totalErrors} errors");
            if (Application.isBatchMode) EditorApplication.Exit(1);
        }
    }
}
