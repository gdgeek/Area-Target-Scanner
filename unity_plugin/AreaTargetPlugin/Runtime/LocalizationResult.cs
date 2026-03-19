using UnityEngine;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public class LocalizationResult : ILocalizationResult
    {
        public bool Success { get; set; }
        public int MapId { get; set; }
        public Matrix4x4 Pose { get; set; }

        public static LocalizationResult Failed()
            => new LocalizationResult { Success = false, MapId = -1, Pose = Matrix4x4.identity };
    }
}
