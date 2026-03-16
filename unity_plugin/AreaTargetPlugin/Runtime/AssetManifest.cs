using System;
using UnityEngine;

namespace AreaTargetPlugin
{
    /// <summary>
    /// Represents the manifest.json file in an area target asset bundle.
    /// </summary>
    [Serializable]
    public class AssetManifest
    {
        public string version;
        public string name;
        public string meshFile;
        public string textureFile;
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
