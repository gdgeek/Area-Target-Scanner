using UnityEngine;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public interface ILocalizationResult
    {
        bool Success { get; }
        int MapId { get; }
        Matrix4x4 Pose { get; }
    }
}
