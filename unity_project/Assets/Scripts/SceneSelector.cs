using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

/// <summary>
/// 场景选择器：启动场景，提供切换到不同测试场景的入口。
/// </summary>
public class SceneSelector : MonoBehaviour
{
    [SerializeField] private Button editorTestButton;
    [SerializeField] private Button arTestButton;
    [SerializeField] private Text infoText;

    void Start()
    {
        if (infoText != null)
        {
            infoText.text = $"AreaTarget Plugin 测试工程\n" +
                            $"平台: {Application.platform}\n" +
                            $"设备: {SystemInfo.deviceModel}\n" +
                            $"Unity: {Application.unityVersion}";
        }

        if (editorTestButton != null)
            editorTestButton.onClick.AddListener(() => SceneManager.LoadScene("TestScene"));

        if (arTestButton != null)
            arTestButton.onClick.AddListener(() => SceneManager.LoadScene("ARTestScene"));
    }
}
