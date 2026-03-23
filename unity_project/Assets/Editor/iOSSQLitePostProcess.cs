#if UNITY_IOS
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEditor.iOS.Xcode;
using System.IO;

/// <summary>
/// iOS 构建后处理：自动添加系统 libsqlite3.tbd 链接。
/// SQLitePCLRaw.provider.e_sqlite3 在 iOS 上通过 __Internal P/Invoke 调用
/// 标准 sqlite3_* 函数，需要系统 SQLite3 库提供这些符号。
/// </summary>
public class iOSSQLitePostProcess
{
    [PostProcessBuild(999)]
    public static void OnPostProcessBuild(BuildTarget target, string path)
    {
        if (target != BuildTarget.iOS) return;

        string projPath = PBXProject.GetPBXProjectPath(path);
        var proj = new PBXProject();
        proj.ReadFromFile(projPath);

        // UnityFramework target 是插件代码实际链接的地方
        string frameworkGuid = proj.GetUnityFrameworkTargetGuid();
        proj.AddFrameworkToProject(frameworkGuid, "libsqlite3.tbd", false);

        // 主 target 也加上以防万一
        string mainGuid = proj.GetUnityMainTargetGuid();
        proj.AddFrameworkToProject(mainGuid, "libsqlite3.tbd", false);

        proj.WriteToFile(projPath);
        UnityEngine.Debug.Log("[iOSSQLitePostProcess] 已添加 libsqlite3.tbd");
    }
}
#endif
