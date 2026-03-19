namespace AreaTargetPlugin.PointCloudLocalization
{
    public class PlatformUpdateResult : IPlatformUpdateResult
    {
        public bool Success { get; set; }
        public int TrackingQuality { get; set; }
        public ICameraData CameraData { get; set; }
    }
}
