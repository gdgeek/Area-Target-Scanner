using System;
using System.Threading.Tasks;
using UnityEngine;

namespace AreaTargetPlugin.PointCloudLocalization
{
    /// <summary>
    /// Default ILocalizer implementation that wraps VisualLocalizationEngine and FeatureDatabaseReader.
    /// Converts ICameraData to CameraFrame, runs ProcessFrame, and converts TrackingResult to ILocalizationResult.
    /// </summary>
    public class PointCloudLocalizer : ILocalizer
    {
        private VisualLocalizationEngine _engine;
        private FeatureDatabaseReader _featureDb;
        private readonly int _mapId;
        private bool _disposed;

        public event Action<int[]> OnSuccessfulLocalizations;

        /// <summary>
        /// Creates a PointCloudLocalizer with pre-initialized engine and feature database.
        /// </summary>
        /// <param name="mapId">The map identifier for this localizer.</param>
        /// <param name="engine">An initialized VisualLocalizationEngine.</param>
        /// <param name="featureDb">A loaded FeatureDatabaseReader.</param>
        public PointCloudLocalizer(int mapId, VisualLocalizationEngine engine, FeatureDatabaseReader featureDb)
        {
            _mapId = mapId;
            _engine = engine;
            _featureDb = featureDb;
        }

        /// <summary>
        /// Performs point cloud localization on the given camera data.
        /// Returns Failed() for null input, invalid dimensions, disposed state, or internal exceptions.
        /// Fires OnSuccessfulLocalizations when tracking succeeds.
        /// </summary>
        public Task<ILocalizationResult> Localize(ICameraData cameraData)
        {
            try
            {
                if (_disposed)
                    return Task.FromResult<ILocalizationResult>(LocalizationResult.Failed());

                if (cameraData == null)
                    return Task.FromResult<ILocalizationResult>(LocalizationResult.Failed());

                if (cameraData.Width <= 0 || cameraData.Height <= 0)
                    return Task.FromResult<ILocalizationResult>(LocalizationResult.Failed());

                var frame = CameraDataAdapter.ToCameraFrame(cameraData);
                var trackingResult = _engine.ProcessFrame(frame);

                if (trackingResult.State == TrackingState.TRACKING)
                {
                    var result = new LocalizationResult
                    {
                        Success = true,
                        MapId = _mapId,
                        Pose = trackingResult.Pose
                    };
                    OnSuccessfulLocalizations?.Invoke(new[] { _mapId });
                    return Task.FromResult<ILocalizationResult>(result);
                }

                return Task.FromResult<ILocalizationResult>(LocalizationResult.Failed());
            }
            catch (Exception ex)
            {
                Debug.LogError($"[PointCloudLocalizer] Localize exception: {ex.Message}");
                return Task.FromResult<ILocalizationResult>(LocalizationResult.Failed());
            }
        }

        /// <summary>
        /// Releases VisualLocalizationEngine and FeatureDatabaseReader resources and marks this localizer as disposed.
        /// After calling this, all subsequent Localize calls return Failed().
        /// Each resource is disposed in its own try-catch to ensure one failure doesn't prevent the other from being released.
        /// The _disposed flag is set before disposal so concurrent Localize calls see it immediately.
        /// </summary>
        public Task StopAndCleanUp()
        {
            if (_disposed) return Task.CompletedTask;
            _disposed = true;

            try
            {
                _engine?.Dispose();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[PointCloudLocalizer] Error disposing engine: {ex.Message}");
            }
            _engine = null;

            try
            {
                _featureDb?.Dispose();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[PointCloudLocalizer] Error disposing feature database: {ex.Message}");
            }
            _featureDb = null;

            return Task.CompletedTask;
        }
    }
}
