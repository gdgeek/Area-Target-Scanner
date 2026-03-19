using UnityEditor;
using UnityEngine;
using System.IO;
using System.Collections.Generic;

public static class ExportPackage
{
    [MenuItem("Build/Export AreaTargetPlugin Package")]
    public static void Export()
    {
        string packageRoot = "Packages/com.areatarget.tracking";
        string outputPath = Path.GetFullPath(Path.Combine(Application.dataPath, "../../unity_plugin/AreaTargetPlugin/AreaTargetPlugin-1.1.0.unitypackage"));

        // Collect all assets under the package
        var assetPaths = new List<string>();

        // Runtime
        CollectAssets(packageRoot + "/Runtime", assetPaths);
        // Tests
        CollectAssets(packageRoot + "/Tests", assetPaths);
        // Editor
        CollectAssets(packageRoot + "/Editor", assetPaths);
        // Root files
        string[] rootFiles = new string[]
        {
            packageRoot + "/package.json",
            packageRoot + "/README.md",
            packageRoot + "/CHANGELOG.md",
            packageRoot + "/LICENSE.md",
            packageRoot + "/TEST_GUIDE.md",
        };
        foreach (var f in rootFiles)
        {
            if (File.Exists(Path.GetFullPath(f)) || AssetDatabase.LoadAssetAtPath<Object>(f) != null)
                assetPaths.Add(f);
        }

        // Also include native plugins
        CollectAssets("Assets/Plugins", assetPaths);

        Debug.Log($"[ExportPackage] Exporting {assetPaths.Count} assets to {outputPath}");

        AssetDatabase.ExportPackage(
            assetPaths.ToArray(),
            outputPath,
            ExportPackageOptions.Recurse
        );

        Debug.Log($"[ExportPackage] Done: {outputPath}");
    }

    private static void CollectAssets(string folder, List<string> results)
    {
        string[] guids = AssetDatabase.FindAssets("", new[] { folder });
        foreach (string guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (!string.IsNullOrEmpty(path))
                results.Add(path);
        }
    }
}
