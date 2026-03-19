using UnityEditor;
using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Linq;

public static class PackageExporter
{
    [MenuItem("Tools/Export AreaTargetPlugin Package")]
    public static void Export()
    {
        // Resolve the local package path
        var packagePath = "Packages/com.areatarget.tracking";
        var outputPath = Path.GetFullPath(
            Path.Combine(Application.dataPath, "../../unity_plugin/AreaTargetPlugin/AreaTargetPlugin-1.2.0.unitypackage")
        );

        // Collect all assets under the package
        var assetPaths = new List<string>();
        CollectAssets(packagePath + "/Runtime", assetPaths);
        CollectAssets(packagePath + "/Editor", assetPaths);
        CollectAssets(packagePath + "/Tests", assetPaths);

        // Include root files
        string[] rootFiles = {
            "package.json", "CHANGELOG.md", "LICENSE.md", "README.md",
            "TEST_GUIDE.md", "Runtime.meta", "Editor.meta", "Tests.meta"
        };
        foreach (var f in rootFiles)
        {
            var p = packagePath + "/" + f;
            if (File.Exists(Path.GetFullPath(p)) || AssetDatabase.LoadAssetAtPath<Object>(p) != null)
                assetPaths.Add(p);
        }

        if (assetPaths.Count == 0)
        {
            Debug.LogError("[PackageExporter] No assets found to export.");
            EditorApplication.Exit(1);
            return;
        }

        Debug.Log($"[PackageExporter] Exporting {assetPaths.Count} assets to {outputPath}");
        AssetDatabase.ExportPackage(
            assetPaths.ToArray(),
            outputPath,
            ExportPackageOptions.Recurse
        );
        Debug.Log($"[PackageExporter] Done: {outputPath}");

        // Exit if running in batch mode
        if (Application.isBatchMode)
            EditorApplication.Exit(0);
    }

    static void CollectAssets(string folder, List<string> results)
    {
        var guids = AssetDatabase.FindAssets("", new[] { folder });
        foreach (var guid in guids)
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            if (!string.IsNullOrEmpty(path))
                results.Add(path);
        }
    }
}
