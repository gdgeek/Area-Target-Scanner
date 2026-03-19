using UnityEditor;
using UnityEngine;
using System.IO;
using System.Collections.Generic;

public static class PackageExporter
{
    [MenuItem("Tools/Export AreaTargetPlugin Package")]
    public static void Export()
    {
        var outputPath = Path.GetFullPath(
            Path.Combine(Application.dataPath, "../../unity_plugin/AreaTargetPlugin/AreaTargetPlugin-1.2.0.unitypackage")
        );

        var assetPaths = new List<string>();

        // 1. Package source code (Runtime, Editor, Tests)
        CollectAssets("Packages/com.areatarget.tracking/Runtime", assetPaths);
        CollectAssets("Packages/com.areatarget.tracking/Editor", assetPaths);
        CollectAssets("Packages/com.areatarget.tracking/Tests", assetPaths);

        // Package root files
        string[] rootFiles = {
            "package.json", "CHANGELOG.md", "LICENSE.md", "README.md",
            "TEST_GUIDE.md", "Runtime.meta", "Editor.meta", "Tests.meta"
        };
        foreach (var f in rootFiles)
        {
            var p = "Packages/com.areatarget.tracking/" + f;
            if (AssetDatabase.LoadAssetAtPath<Object>(p) != null)
                assetPaths.Add(p);
        }

        // 2. Native plugins (dylib, so, dll) and managed DLLs
        CollectAssets("Assets/Plugins", assetPaths);

        // 3. link.xml (prevents IL2CPP from stripping SQLite assemblies)
        if (AssetDatabase.LoadAssetAtPath<Object>("Assets/link.xml") != null)
            assetPaths.Add("Assets/link.xml");

        if (assetPaths.Count == 0)
        {
            Debug.LogError("[PackageExporter] No assets found to export.");
            if (Application.isBatchMode) EditorApplication.Exit(1);
            return;
        }

        Debug.Log($"[PackageExporter] Exporting {assetPaths.Count} assets to {outputPath}");
        AssetDatabase.ExportPackage(
            assetPaths.ToArray(),
            outputPath,
            ExportPackageOptions.Recurse
        );
        Debug.Log($"[PackageExporter] Done: {outputPath}");

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
