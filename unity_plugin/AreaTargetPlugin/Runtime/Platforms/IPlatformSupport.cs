using System.Threading.Tasks;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public interface IPlatformSupport
    {
        Task<IPlatformUpdateResult> UpdatePlatform();
        Task ConfigurePlatform();
        Task StopAndCleanUp();
    }
}
