using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace AreaTargetPlugin.Tests
{
    [TestFixture]
    public class AreaTargetTrackerInitTests
    {
        private string _testDir;

        [SetUp]
        public void SetUp()
        {
            _testDir = Path.Combine(Path.GetTempPath(), "TrackerInitTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_testDir);
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_testDir))
            {
                Directory.Delete(_testDir, true);
            }
        }

        private string CreateValidAssetBundle()
        {
            string manifest = @"{
                ""version"": ""1.0"",
                ""name"": ""test_asset"",
                ""meshFile"": ""mesh.obj"",
                ""textureFile"": ""texture_atlas.png"",
                ""featureDbFile"": ""features.db"",
                ""keyframeCount"": 5,
                ""featureType"": ""ORB"",
                ""createdAt"": ""2024-01-01T00:00:00Z""
            }";
            File.WriteAllText(Path.Combine(_testDir, "manifest.json"), manifest);
            File.WriteAllText(Path.Combine(_testDir, "mesh.obj"), "# OBJ");
            File.WriteAllText(Path.Combine(_testDir, "texture_atlas.png"), "PNG");
            File.WriteAllText(Path.Combine(_testDir, "features.db"), "DB");
            return _testDir;
        }

        [Test]
        public void Initialize_ValidAsset_ReturnsTrue()
        {
            string assetDir = CreateValidAssetBundle();
            var tracker = new AreaTargetTracker();

            bool result = tracker.Initialize(assetDir);

            Assert.IsTrue(result);
            tracker.Dispose();
        }

        [Test]
        public void Initialize_ValidAsset_SetsStateToInitializing()
        {
            string assetDir = CreateValidAssetBundle();
            var tracker = new AreaTargetTracker();

            tracker.Initialize(assetDir);

            Assert.AreEqual(TrackingState.INITIALIZING, tracker.GetTrackingState());
            tracker.Dispose();
        }

        [Test]
        public void Initialize_InvalidPath_ReturnsFalse()
        {
            var tracker = new AreaTargetTracker();

            bool result = tracker.Initialize("/nonexistent/path");

            Assert.IsFalse(result);
            tracker.Dispose();
        }

        [Test]
        public void Initialize_NullPath_ReturnsFalse()
        {
            var tracker = new AreaTargetTracker();

            bool result = tracker.Initialize(null);

            Assert.IsFalse(result);
            tracker.Dispose();
        }

        [Test]
        public void Initialize_CorruptManifest_ReturnsFalse()
        {
            Directory.CreateDirectory(_testDir);
            File.WriteAllText(Path.Combine(_testDir, "manifest.json"), "CORRUPT DATA");

            var tracker = new AreaTargetTracker();
            bool result = tracker.Initialize(_testDir);

            Assert.IsFalse(result);
            tracker.Dispose();
        }

        [Test]
        public void Initialize_IncompatibleVersion_ReturnsFalse()
        {
            string manifest = @"{
                ""version"": ""99.0"",
                ""name"": ""test"",
                ""meshFile"": ""mesh.obj"",
                ""textureFile"": ""texture_atlas.png"",
                ""featureDbFile"": ""features.db""
            }";
            File.WriteAllText(Path.Combine(_testDir, "manifest.json"), manifest);
            File.WriteAllText(Path.Combine(_testDir, "mesh.obj"), "data");
            File.WriteAllText(Path.Combine(_testDir, "texture_atlas.png"), "data");
            File.WriteAllText(Path.Combine(_testDir, "features.db"), "data");

            var tracker = new AreaTargetTracker();
            bool result = tracker.Initialize(_testDir);

            Assert.IsFalse(result);
            tracker.Dispose();
        }

        [Test]
        public void Initialize_AfterDispose_ReturnsFalse()
        {
            string assetDir = CreateValidAssetBundle();
            var tracker = new AreaTargetTracker();
            tracker.Dispose();

            bool result = tracker.Initialize(assetDir);

            Assert.IsFalse(result);
        }

        [Test]
        public void ProcessFrame_BeforeInitialize_ReturnsLost()
        {
            var tracker = new AreaTargetTracker();
            var frame = new CameraFrame
            {
                ImageData = new byte[100],
                Width = 10,
                Height = 10,
                Intrinsics = Matrix4x4.identity
            };

            TrackingResult result = tracker.ProcessFrame(frame);

            Assert.AreEqual(TrackingState.LOST, result.State);
            Assert.AreEqual(0f, result.Confidence);
            Assert.AreEqual(0, result.MatchedFeatures);
            tracker.Dispose();
        }
    }
}
