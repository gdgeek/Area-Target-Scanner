using System;
using UnityEngine;

namespace AreaTargetPlugin
{
    /// <summary>
    /// Represents the manifest.json file in an area target asset bundle (v2.0 format).
    /// v2.0 uses GLB format (meshFile points to .glb), textureFile is no longer required
    /// as textures are embedded in the GLB file.
    /// </summary>
    [Serializable]
    public class AssetManifest
    {
        public string version;
        public string name;
        public string meshFile;
        public string format;
        public string textureFile;  // Retained for backward compatibility; not required in v2.0
        public string featureDbFile;
        public BoundsData bounds;
        public int keyframeCount;
        public string featureType;
        public string createdAt;
    }

    /// <summary>
    /// Bounding box data from the manifest.
    /// </summary>
    [Serializable]
    public class BoundsData
    {
        public float[] min;
        public float[] max;
    }
}
