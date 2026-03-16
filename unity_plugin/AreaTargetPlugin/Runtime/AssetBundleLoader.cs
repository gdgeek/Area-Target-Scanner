using System;
using System.IO;
using UnityEngine;

namespace AreaTargetPlugin
{
    /// <summary>
    /// Loads and validates area target asset bundles from disk.
    /// Expected asset bundle structure:
    ///   manifest.json, mesh.obj, texture_atlas.png, features.db
    /// </summary>
    public class AssetBundleLoader
    {
        private const string SupportedVersion = "1.0";

        /// <summary>The parsed manifest after a successful load.</summary>
        public AssetManifest Manifest { get; private set; }

        /// <summary>Full path to the loaded mesh file.</summary>
        public string MeshPath { get; private set; }

        /// <summary>Full path to the loaded texture file.</summary>
        public string TexturePath { get; private set; }

        /// <summary>Full path to the loaded feature database file.</summary>
        public string FeatureDbPath { get; private set; }

        /// <summary>Error message from the last failed load attempt.</summary>
        public string LastError { get; private set; }

        /// <summary>
        /// Loads the area target asset bundle at the given path.
        /// Parses manifest.json, validates completeness and version compatibility,
        /// and verifies that all referenced files exist.
        /// </summary>
        /// <param name="assetPath">Directory containing the asset bundle files.</param>
        /// <returns>True if all assets loaded and validated successfully.</returns>
        public bool Load(string assetPath)
        {
            LastError = null;
            Manifest = null;
            MeshPath = null;
            TexturePath = null;
            FeatureDbPath = null;

            if (string.IsNullOrEmpty(assetPath))
            {
                LastError = "Asset path is null or empty.";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            if (!Directory.Exists(assetPath))
            {
                LastError = $"Asset directory does not exist: {assetPath}";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            // Load and parse manifest.json
            string manifestPath = Path.Combine(assetPath, "manifest.json");
            if (!File.Exists(manifestPath))
            {
                LastError = $"manifest.json not found in: {assetPath}";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            string manifestJson;
            try
            {
                manifestJson = File.ReadAllText(manifestPath);
            }
            catch (Exception ex)
            {
                LastError = $"Failed to read manifest.json: {ex.Message}";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            AssetManifest manifest;
            try
            {
                manifest = JsonUtility.FromJson<AssetManifest>(manifestJson);
            }
            catch (Exception ex)
            {
                LastError = $"Failed to parse manifest.json: {ex.Message}";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            if (manifest == null)
            {
                LastError = "manifest.json parsed to null.";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            // Validate version compatibility
            if (string.IsNullOrEmpty(manifest.version))
            {
                LastError = "manifest.json missing 'version' field.";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            if (!manifest.version.StartsWith(SupportedVersion))
            {
                LastError = $"Incompatible asset version: {manifest.version} (supported: {SupportedVersion}).";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            // Validate required manifest fields
            if (string.IsNullOrEmpty(manifest.meshFile))
            {
                LastError = "manifest.json missing 'meshFile' field.";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            if (string.IsNullOrEmpty(manifest.textureFile))
            {
                LastError = "manifest.json missing 'textureFile' field.";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            if (string.IsNullOrEmpty(manifest.featureDbFile))
            {
                LastError = "manifest.json missing 'featureDbFile' field.";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            // Verify referenced files exist
            string meshPath = Path.Combine(assetPath, manifest.meshFile);
            if (!File.Exists(meshPath))
            {
                LastError = $"Mesh file not found: {manifest.meshFile}";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            string texturePath = Path.Combine(assetPath, manifest.textureFile);
            if (!File.Exists(texturePath))
            {
                LastError = $"Texture file not found: {manifest.textureFile}";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            string featureDbPath = Path.Combine(assetPath, manifest.featureDbFile);
            if (!File.Exists(featureDbPath))
            {
                LastError = $"Feature database not found: {manifest.featureDbFile}";
                Debug.LogError($"[AreaTargetPlugin] {LastError}");
                return false;
            }

            // All validations passed
            Manifest = manifest;
            MeshPath = meshPath;
            TexturePath = texturePath;
            FeatureDbPath = featureDbPath;

            Debug.Log($"[AreaTargetPlugin] Asset bundle loaded: {manifest.name} v{manifest.version} " +
                      $"({manifest.keyframeCount} keyframes, feature type: {manifest.featureType})");
            return true;
        }
    }
}
