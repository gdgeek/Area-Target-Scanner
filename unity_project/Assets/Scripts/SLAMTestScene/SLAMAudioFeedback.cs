using UnityEngine;

/// <summary>
/// 管理跟踪状态转换时的音效播放。
/// 使用 PlayOneShot 播放音效，避免中断正在播放的音效。
/// AudioClip 或 AudioSource 未赋值时静默跳过，不报错。
/// </summary>
public class SLAMAudioFeedback : MonoBehaviour
{
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip trackingFoundClip;
    [SerializeField] private AudioClip trackingLostClip;

    /// <summary>
    /// 播放跟踪成功音效。AudioSource 或 AudioClip 为 null 时静默跳过。
    /// </summary>
    public void PlayTrackingFound()
    {
        if (audioSource != null && trackingFoundClip != null)
        {
            audioSource.PlayOneShot(trackingFoundClip);
        }
    }

    /// <summary>
    /// 播放跟踪丢失音效。AudioSource 或 AudioClip 为 null 时静默跳过。
    /// </summary>
    public void PlayTrackingLost()
    {
        if (audioSource != null && trackingLostClip != null)
        {
            audioSource.PlayOneShot(trackingLostClip);
        }
    }
}
