using System;
using System.Threading.Tasks;
using UnityEngine;

namespace AreaTargetPlugin.PointCloudLocalization
{
    /// <summary>
    /// AR Foundation-based IPlatformSupport implementation.
    /// Uses ARCameraManager for camera frames, XRCameraSubsystem for intrinsics,
    /// and XROrigin for tracking pose. Compatible with ARKit XR Plugin (iOS) and
    /// OpenXR Plugin (Rokid/Pico/Quest) without platform-specific code branches.
    /// Requires com.unity.xr.arfoundation package.
    ///
    /// Coordinate system conversion is performed before returning ICameraData,
    /// ensuring all poses are in Unity left-handed Y-up coordinate system.
    /// </summary>
    public class ARFoundationPlatformSupport : IPlatformSupport
    {
        private bool _configured;
        private bool _disposed;

#if UNITY_AR_FOUNDATION
        // AR Foundation references (resolved during ConfigurePlatform)
        private UnityEngine.XR.ARFoundation.ARCameraManager _cameraManager;
        private Unity.XR.CoreUtils.XROrigin _xrOrigin;
#endif

        public async Task<IPlatformUpdateResult> UpdatePlatform()
        {
            if (_disposed || !_configured)
            {
                return new PlatformUpdateResult
                {
                    Success = false,
                    TrackingQuality = 0,
                    CameraData = null
                };
            }

#if UNITY_AR_FOUNDATION
            return await AcquireFrameFromARFoundation();
#else
            await Task.CompletedTask;
            Debug.LogWarning(
                "[ARFoundationPlatformSupport] AR Foundation package not available. " +
                "Install com.unity.xr.arfoundation and add UNITY_AR_FOUNDATION to scripting defines.");
            return new PlatformUpdateResult
            {
                Success = false,
                TrackingQuality = 0,
                CameraData = null
            };
#endif
        }

        public Task ConfigurePlatform()
        {
            if (_disposed)
            {
                Debug.LogError("[ARFoundationPlatformSupport] Cannot configure after disposal.");
                return Task.CompletedTask;
            }

#if UNITY_AR_FOUNDATION
            ConfigureARFoundation();
#else
            Debug.LogWarning(
                "[ARFoundationPlatformSupport] AR Foundation package not available. " +
                "Skipping platform configuration.");
#endif
            _configured = true;
            return Task.CompletedTask;
        }

        public Task StopAndCleanUp()
        {
            if (_disposed) return Task.CompletedTask;

            _disposed = true;
            _configured = false;

#if UNITY_AR_FOUNDATION
            CleanUpARFoundation();
#endif
            return Task.CompletedTask;
        }

#if UNITY_AR_FOUNDATION
        private void ConfigureARFoundation()
        {
            // Find ARCameraManager in scene — works for both ARKit and OpenXR providers
            _cameraManager = UnityEngine.Object.FindAnyObjectByType<UnityEngine.XR.ARFoundation.ARCameraManager>();
            if (_cameraManager == null)
            {
                Debug.LogError("[ARFoundationPlatformSupport] ARCameraManager not found in scene.");
                return;
            }

            // Find XROrigin — unified tracking root for all AR Foundation providers
            _xrOrigin = UnityEngine.Object.FindAnyObjectByType<Unity.XR.CoreUtils.XROrigin>();
            if (_xrOrigin == null)
            {
                Debug.LogError("[ARFoundationPlatformSupport] XROrigin not found in scene.");
                return;
            }
        }

        private async Task<IPlatformUpdateResult> AcquireFrameFromARFoundation()
        {
            // 1. Acquire latest CPU image from ARCameraManager
            //    This API is provider-agnostic: works with ARKit XR Plugin and OpenXR Plugin
            if (!_cameraManager.TryAcquireLatestCpuImage(out var cpuImage))
            {
                return new PlatformUpdateResult { Success = false, TrackingQuality = 0, CameraData = null };
            }

            try
            {
                // 2. Get camera intrinsics from XRCameraSubsystem
                //    AR Foundation abstracts ARKit/OpenXR intrinsics into a unified XRCameraIntrinsics
                var subsystem = _cameraManager.subsystem;
                if (subsystem == null || !subsystem.TryGetIntrinsics(out var intrinsics))
                {
                    return new PlatformUpdateResult { Success = false, TrackingQuality = 0, CameraData = null };
                }

                // 3. Get tracking pose from XROrigin camera
                //    XROrigin.Camera provides the tracked camera transform in Unity world space
                var cameraTransform = _xrOrigin.Camera.transform;
                var position = cameraTransform.position;
                var rotation = cameraTransform.rotation;

                // 4. Convert CPU image to byte array (grayscale preferred for feature extraction)
                var conversionParams = new UnityEngine.XR.ARFoundation.XRCpuImage.ConversionParams
                {
                    inputRect = new RectInt(0, 0, cpuImage.width, cpuImage.height),
                    outputDimensions = new Vector2Int(cpuImage.width, cpuImage.height),
                    outputFormat = UnityEngine.TextureFormat.R8,
                    transformation = UnityEngine.XR.ARFoundation.XRCpuImage.Transformation.None
                };

                int bufferSize = cpuImage.GetConvertedDataSize(conversionParams);
                var imageBytes = new byte[bufferSize];

                unsafe
                {
                    fixed (byte* ptr = imageBytes)
                    {
                        cpuImage.Convert(conversionParams, (IntPtr)ptr, bufferSize);
                    }
                }

                // 5. Coordinate system conversion:
                //    AR Foundation already provides poses in Unity left-handed Y-up coordinate system
                //    for both ARKit and OpenXR providers. No additional conversion needed here
                //    because XROrigin handles the provider-to-Unity transform internally.
                //
                //    If a non-AR-Foundation platform (e.g., Rokid UXR SDK) uses a different
                //    coordinate convention, a custom IPlatformSupport implementation should
                //    handle the conversion before returning ICameraData.

                // 6. Assess tracking quality from ARSession state
                int trackingQuality = EvaluateTrackingQuality();

                var cameraData = new ARFoundationCameraData(
                    imageBytes,
                    cpuImage.width,
                    cpuImage.height,
                    channels: 1, // grayscale
                    new Vector4(intrinsics.focalLength.x, intrinsics.focalLength.y,
                                intrinsics.principalPoint.x, intrinsics.principalPoint.y),
                    position,
                    rotation
                );

                return new PlatformUpdateResult
                {
                    Success = true,
                    TrackingQuality = trackingQuality,
                    CameraData = cameraData
                };
            }
            finally
            {
                cpuImage.Dispose();
            }
        }

        private int EvaluateTrackingQuality()
        {
            // Map ARSession tracking state to 0-100 quality score
            // This works identically for ARKit and OpenXR providers
            var state = UnityEngine.XR.ARFoundation.ARSession.state;
            switch (state)
            {
                case UnityEngine.XR.ARFoundation.ARSessionState.SessionTracking:
                    return 100;
                case UnityEngine.XR.ARFoundation.ARSessionState.SessionInitializing:
                    return 30;
                case UnityEngine.XR.ARFoundation.ARSessionState.Ready:
                    return 10;
                default:
                    return 0;
            }
        }

        private void CleanUpARFoundation()
        {
            _cameraManager = null;
            _xrOrigin = null;
        }

        /// <summary>
        /// Internal ICameraData implementation for AR Foundation frames.
        /// </summary>
        private class ARFoundationCameraData : ICameraData
        {
            private readonly byte[] _bytes;
            public int Width { get; }
            public int Height { get; }
            public int Channels { get; }
            public Vector4 Intrinsics { get; }
            public Vector3 CameraPositionOnCapture { get; }
            public Quaternion CameraRotationOnCapture { get; }

            public ARFoundationCameraData(
                byte[] bytes, int width, int height, int channels,
                Vector4 intrinsics, Vector3 position, Quaternion rotation)
            {
                _bytes = bytes;
                Width = width;
                Height = height;
                Channels = channels;
                Intrinsics = intrinsics;
                CameraPositionOnCapture = position;
                CameraRotationOnCapture = rotation;
            }

            public byte[] GetBytes() => _bytes;
        }
#endif
    }
}
