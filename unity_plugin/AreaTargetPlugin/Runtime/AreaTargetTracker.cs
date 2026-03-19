using System;
using UnityEngine;

namespace AreaTargetPlugin
{
    /// <summary>
    /// Main implementation of the area target tracker.
    /// Loads asset bundles and provides 6DoF visual localization.
    /// Chains VisualLocalizationEngine → KalmanPoseFilter for smooth tracking.
    /// </summary>
    /// <remarks>
    /// Privacy: No network permissions required. Camera data is processed locally only.
    /// This class does not use HttpClient, WebRequest, or any networking APIs.
    /// Camera access is only used during active tracking via ProcessFrame.
    /// Requirements: 15.3, 15.4
    /// </remarks>
    public class AreaTargetTracker : IAreaTargetTracker
    {
        private TrackingState _state = TrackingState.INITIALIZING;
        private AssetBundleLoader _loader;
        private VisualLocalizationEngine _localizationEngine;
        private KalmanPoseFilter _kalmanFilter;
        private FeatureDatabaseReader _featureDb;
        private bool _initialized;
        private bool _disposed;

        public AreaTargetTracker()
        {
            _loader = new AssetBundleLoader();
        }

        /// <inheritdoc/>
        public bool Initialize(string assetPath)
        {
            if (_disposed)
            {
                Debug.LogError("[AreaTargetPlugin] Cannot initialize a disposed tracker.");
                return false;
            }

            bool success = _loader.Load(assetPath);
            if (!success)
            {
                return false;
            }

            // Load the feature database from the asset bundle
            _featureDb = new FeatureDatabaseReader();
            if (!_featureDb.Load(_loader.FeatureDbPath))
            {
                Debug.LogError("[AreaTargetPlugin] Failed to load feature database.");
                _featureDb = null;
                return false;
            }

            // Initialize the visual localization engine
            _localizationEngine = new VisualLocalizationEngine();
            if (!_localizationEngine.Initialize(_featureDb))
            {
                Debug.LogError("[AreaTargetPlugin] Failed to initialize localization engine.");
                _localizationEngine.Dispose();
                _localizationEngine = null;
                _featureDb.Dispose();
                _featureDb = null;
                return false;
            }

            // Initialize the Kalman pose filter for smoothing
            _kalmanFilter = new KalmanPoseFilter();

            _initialized = true;
            _state = TrackingState.INITIALIZING;
            Debug.Log("[AreaTargetPlugin] Tracker initialized successfully.");
            return true;
        }

        /// <inheritdoc/>
        public TrackingResult ProcessFrame(CameraFrame cameraFrame)
        {
            if (!_initialized || _disposed)
            {
                return new TrackingResult
                {
                    State = TrackingState.LOST,
                    Pose = Matrix4x4.identity,
                    Confidence = 0f,
                    MatchedFeatures = 0
                };
            }

            // Step 1: Run visual localization engine (ORB → BoW → match → PnP)
            TrackingResult locResult = _localizationEngine.ProcessFrame(cameraFrame);

            // Step 2: Apply Kalman filter smoothing when tracking
            if (locResult.State == TrackingState.TRACKING)
            {
                Matrix4x4 smoothedPose = _kalmanFilter.Update(locResult.Pose);
                _state = TrackingState.TRACKING;

                return new TrackingResult
                {
                    State = TrackingState.TRACKING,
                    Pose = smoothedPose,
                    Confidence = locResult.Confidence,
                    MatchedFeatures = locResult.MatchedFeatures
                };
            }

            // LOST or INITIALIZING — pass through without smoothing
            _state = locResult.State == TrackingState.LOST ? TrackingState.LOST : _state;
            return locResult;
        }

        /// <inheritdoc/>
        public TrackingState GetTrackingState()
        {
            return _state;
        }

        /// <inheritdoc/>
        /// <remarks>
        /// Clears tracking state, resets the Kalman filter and localization engine,
        /// and restarts localization from scratch.
        /// Validates: Requirements 14.4
        /// </remarks>
        public void Reset()
        {
            _state = TrackingState.INITIALIZING;

            _kalmanFilter?.Reset();
            _localizationEngine?.ResetState();

            Debug.Log("[AreaTargetPlugin] Tracker reset.");
        }

        /// <inheritdoc/>
        /// <remarks>
        /// Releases all resources: localization engine, feature database,
        /// asset loader, and Kalman filter.
        /// Validates: Requirements 14.5
        /// </remarks>
        public void Dispose()
        {
            if (_disposed) return;

            _disposed = true;
            _initialized = false;

            _localizationEngine?.Dispose();
            _localizationEngine = null;

            _featureDb?.Dispose();
            _featureDb = null;

            _loader = null;
            _kalmanFilter = null;

            _state = TrackingState.LOST;
            Debug.Log("[AreaTargetPlugin] Tracker disposed.");
        }
    }
}
