namespace AreaTargetPlugin.PointCloudLocalization
{
    public interface IPlatformUpdateResult
    {
        bool Success { get; }
        int TrackingQuality { get; }  // 0-100
        ICameraData CameraData { get; }
    }
}
