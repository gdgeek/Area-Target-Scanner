using System;
using System.Threading.Tasks;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public interface ILocalizer
    {
        event Action<int[]> OnSuccessfulLocalizations;
        Task<ILocalizationResult> Localize(ICameraData cameraData);
        Task StopAndCleanUp();
    }
}
