using System;
using System.Threading.Tasks;
using UnityEngine;

namespace AreaTargetPlugin.PointCloudLocalization
{
    /// <summary>
    /// Orchestrates the end-to-end localization pipeline:
    /// IPlatformSupport.UpdatePlatform() → TrackingQuality threshold check →
    /// ILocalizer.Localize() → MapManager.TryGetMapEntry() →
    /// ISceneUpdater.UpdateScene() → XRSpace.SceneUpdate() → IDataProcessor chain → Transform update.
    /// </summary>
    public class LocalizationPipeline
    {
        private readonly IPlatformSupport _platformSupport;
        private readonly ILocalizer _localizer;
        private readonly ISceneUpdater _sceneUpdater;

        /// <summary>
        /// TrackingQuality threshold (0-100). Frames with quality below this value are skipped.
        /// </summary>
        public int TrackingQualityThreshold { get; set; } = 50;

        /// <summary>
        /// Creates a new LocalizationPipeline with the required dependencies.
        /// </summary>
        /// <param name="platformSupport">Platform abstraction for camera frame acquisition.</param>
        /// <param name="localizer">Localizer that performs point cloud localization.</param>
        /// <param name="sceneUpdater">Scene updater that bridges localization results to XRSpace.</param>
        public LocalizationPipeline(IPlatformSupport platformSupport, ILocalizer localizer, ISceneUpdater sceneUpdater)
        {
            _platformSupport = platformSupport ?? throw new ArgumentNullException(nameof(platformSupport));
            _localizer = localizer ?? throw new ArgumentNullException(nameof(localizer));
            _sceneUpdater = sceneUpdater ?? throw new ArgumentNullException(nameof(sceneUpdater));
        }

        /// <summary>
        /// Executes one frame of the full localization pipeline.
        /// Returns the localization result, or a failed result if the frame was skipped.
        /// </summary>
        public async Task<ILocalizationResult> RunFrame()
        {
            // Step 1: Update platform to get camera data
            IPlatformUpdateResult platformResult;
            try
            {
                platformResult = await _platformSupport.UpdatePlatform();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[LocalizationPipeline] Platform update failed: {ex.Message}");
                return LocalizationResult.Failed();
            }

            // Step 2: Check platform success
            if (!platformResult.Success)
            {
                return LocalizationResult.Failed();
            }

            // Step 3: Check tracking quality threshold
            if (platformResult.TrackingQuality < TrackingQualityThreshold)
            {
                Debug.Log($"[LocalizationPipeline] TrackingQuality {platformResult.TrackingQuality} below threshold {TrackingQualityThreshold}, skipping frame.");
                return LocalizationResult.Failed();
            }

            // Step 4: Run localization
            var cameraData = platformResult.CameraData;
            ILocalizationResult localizationResult;
            try
            {
                localizationResult = await _localizer.Localize(cameraData);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[LocalizationPipeline] Localization failed: {ex.Message}");
                return LocalizationResult.Failed();
            }

            // Step 5: If localization failed, return early
            if (!localizationResult.Success)
            {
                return localizationResult;
            }

            // Step 6: Look up map entry
            if (!MapManager.TryGetMapEntry(localizationResult.MapId, out var mapEntry))
            {
                Debug.LogWarning($"[LocalizationPipeline] Localization succeeded for mapId {localizationResult.MapId} but no MapEntry found. Skipping scene update.");
                return localizationResult;
            }

            // Step 7: Update scene (SceneUpdater → XRSpace.SceneUpdate → IDataProcessor chain → Transform)
            try
            {
                await _sceneUpdater.UpdateScene(mapEntry, cameraData, localizationResult);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[LocalizationPipeline] Scene update failed: {ex.Message}");
            }

            return localizationResult;
        }

        /// <summary>
        /// Stops the pipeline and cleans up platform and localizer resources.
        /// </summary>
        public async Task StopAndCleanUp()
        {
            try
            {
                await _platformSupport.StopAndCleanUp();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[LocalizationPipeline] Error stopping platform: {ex.Message}");
            }

            try
            {
                await _localizer.StopAndCleanUp();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[LocalizationPipeline] Error stopping localizer: {ex.Message}");
            }
        }
    }
}
