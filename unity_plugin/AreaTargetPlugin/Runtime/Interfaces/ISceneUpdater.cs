using System.Threading.Tasks;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public interface ISceneUpdater
    {
        Task UpdateScene(MapEntry entry, ICameraData cameraData, ILocalizationResult result);
    }
}
