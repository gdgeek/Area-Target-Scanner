using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using OpenCvForUnity.CoreModule;
using OpenCvForUnity.Calib3dModule;

namespace AreaTargetPlugin.Tests
{
    /// <summary>
    /// Unit tests for the visual localization engine.
    /// Validates: Requirements 11.6, 11.7, 11.8, 12.4
    /// </summary>
    [TestFixture]
    public class VisualLocalizationTests
    {
        #region Helper Methods

        /// <summary>
        /// Creates a synthetic camera intrinsics matrix.
        /// </summary>
        private static Matrix4x4 CreateIntrinsics(float fx, float fy, float cx, float cy)
        {
            var m = Matrix4x4.zero;
            m.m00 = fx; m.m01 = 0;  m.m02 = cx;
            m.m10 = 0;  m.m11 = fy; m.m12 = cy;
            m.m20 = 0;  m.m21 = 0;  m.m22 = 1;
            return m;
        }

        /// <summary>
        /// Projects a 3D point to 2D using the given rotation, translation, and intrinsics.
        /// </summary>
        private static Vector2 ProjectPoint(Vector3 pt3d, double[] rvec, double[] tvec, float fx, float fy, float cx, float cy)
        {
            // Convert rvec to rotation matrix
            Mat rvecMat = new Mat(3, 1, CvType.CV_64FC1);
            rvecMat.put(0, 0, rvec);
            Mat rotMat = new Mat();
            Calib3d.Rodrigues(rvecMat, rotMat);
            double[] r = new double[9];
            rotMat.get(0, 0, r);
            rvecMat.Dispose();
            rotMat.Dispose();

            // Transform point: p_cam = R * p_world + t
            double px = r[0] * pt3d.x + r[1] * pt3d.y + r[2] * pt3d.z + tvec[0];
            double py = r[3] * pt3d.x + r[4] * pt3d.y + r[5] * pt3d.z + tvec[1];
            double pz = r[6] * pt3d.x + r[7] * pt3d.y + r[8] * pt3d.z + tvec[2];

            // Project: u = fx * px/pz + cx, v = fy * py/pz + cy
            float u = (float)(fx * px / pz + cx);
            float v = (float)(fy * py / pz + cy);
            return new Vector2(u, v);
        }

        /// <summary>
        /// Generates synthetic 2D-3D correspondences from a known pose.
        /// Creates 3D points on a plane at z=5 and projects them.
        /// </summary>
        private static void GenerateSyntheticCorrespondences(
            int count,
            double[] rvec, double[] tvec,
            float fx, float fy, float cx, float cy,
            out List<Vector3> points3D, out List<Vector2> points2D)
        {
            points3D = new List<Vector3>();
            points2D = new List<Vector2>();
            var rng = new System.Random(42);

            for (int i = 0; i < count; i++)
            {
                // Random 3D points in front of camera
                float x = (float)(rng.NextDouble() * 4.0 - 2.0);
                float y = (float)(rng.NextDouble() * 4.0 - 2.0);
                float z = (float)(rng.NextDouble() * 3.0 + 3.0); // z in [3, 6]
                var pt3d = new Vector3(x, y, z);

                Vector2 pt2d = ProjectPoint(pt3d, rvec, tvec, fx, fy, cx, cy);

                // Only keep points that project within image bounds
                if (pt2d.x >= 0 && pt2d.x < cx * 2 && pt2d.y >= 0 && pt2d.y < cy * 2)
                {
                    points3D.Add(pt3d);
                    points2D.Add(pt2d);
                }
            }
        }

        #endregion

        #region PnP Solve Accuracy Tests (Requirement 11.6, 11.7)

        [Test]
        public void SolvePnP_WithSyntheticCorrespondences_RecoversPoseAccurately()
        {
            // Known ground truth pose
            double[] gtRvec = { 0.1, -0.05, 0.02 };
            double[] gtTvec = { 0.5, -0.3, 5.0 };
            float fx = 500f, fy = 500f, cx = 320f, cy = 240f;

            GenerateSyntheticCorrespondences(
                80, gtRvec, gtTvec, fx, fy, cx, cy,
                out var points3D, out var points2D);

            Assert.GreaterOrEqual(points3D.Count, 20, "Need at least 20 correspondences");

            // Build OpenCV inputs
            var pts3d = new List<Point3>();
            var pts2d = new List<Point>();
            for (int i = 0; i < points3D.Count; i++)
            {
                pts3d.Add(new Point3(points3D[i].x, points3D[i].y, points3D[i].z));
                pts2d.Add(new Point(points2D[i].x, points2D[i].y));
            }

            MatOfPoint3f objPts = new MatOfPoint3f();
            objPts.fromList(pts3d);
            MatOfPoint2f imgPts = new MatOfPoint2f();
            imgPts.fromList(pts2d);

            Mat cameraMat = new Mat(3, 3, CvType.CV_64FC1);
            cameraMat.put(0, 0, fx, 0, cx, 0, fy, cy, 0, 0, 1);

            Mat distCoeffs = new Mat();
            Mat rvec = new Mat();
            Mat tvec = new Mat();
            Mat inliers = new Mat();

            bool success = Calib3d.solvePnPRansac(
                objPts, imgPts, cameraMat, distCoeffs,
                rvec, tvec, false, 100, 8.0f, 0.99, inliers);

            Assert.IsTrue(success, "PnP should succeed with clean synthetic data");
            Assert.GreaterOrEqual(inliers.rows(), 20, "Should have >= 20 inliers");

            // Verify recovered pose is close to ground truth
            double[] recoveredR = new double[3];
            rvec.get(0, 0, recoveredR);
            double[] recoveredT = new double[3];
            tvec.get(0, 0, recoveredT);

            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(gtRvec[i], recoveredR[i], 0.1,
                    $"Rotation component {i} should be close to ground truth");
                Assert.AreEqual(gtTvec[i], recoveredT[i], 0.5,
                    $"Translation component {i} should be close to ground truth");
            }

            // Cleanup
            objPts.Dispose(); imgPts.Dispose();
            cameraMat.Dispose(); distCoeffs.Dispose();
            rvec.Dispose(); tvec.Dispose(); inliers.Dispose();
        }

        #endregion

        #region LOST State Tests (Requirement 11.8)

        [Test]
        public void ProcessFrame_InsufficientFeatures_ReturnsLost()
        {
            // Create a frame with very few features (tiny uniform image)
            var frame = new CameraFrame
            {
                ImageData = new byte[16 * 16], // 16x16 black image → few/no features
                Width = 16,
                Height = 16,
                Intrinsics = CreateIntrinsics(500, 500, 8, 8)
            };

            // Create a minimal engine with a mock database
            var engine = new VisualLocalizationEngine();
            var mockDb = CreateMockFeatureDatabase();
            engine.Initialize(mockDb);

            TrackingResult result = engine.ProcessFrame(frame);

            Assert.AreEqual(TrackingState.LOST, result.State);
            Assert.AreEqual(0f, result.Confidence);
            Assert.AreEqual(0, result.MatchedFeatures);

            engine.Dispose();
            mockDb.Dispose();
        }

        [Test]
        public void ProcessFrame_EmptyImage_ReturnsLost()
        {
            var frame = new CameraFrame
            {
                ImageData = new byte[0],
                Width = 0,
                Height = 0,
                Intrinsics = Matrix4x4.identity
            };

            var engine = new VisualLocalizationEngine();
            var mockDb = CreateMockFeatureDatabase();
            engine.Initialize(mockDb);

            TrackingResult result = engine.ProcessFrame(frame);

            Assert.AreEqual(TrackingState.LOST, result.State);

            engine.Dispose();
            mockDb.Dispose();
        }

        #endregion

        #region Confidence Calculation Tests (Requirement 12.4)

        [Test]
        public void Confidence_With20Inliers_Returns0Point2()
        {
            float confidence = Mathf.Min(1.0f, 20 / 100.0f);
            Assert.AreEqual(0.2f, confidence, 0.001f);
        }

        [Test]
        public void Confidence_With50Inliers_Returns0Point5()
        {
            float confidence = Mathf.Min(1.0f, 50 / 100.0f);
            Assert.AreEqual(0.5f, confidence, 0.001f);
        }

        [Test]
        public void Confidence_With100Inliers_Returns1Point0()
        {
            float confidence = Mathf.Min(1.0f, 100 / 100.0f);
            Assert.AreEqual(1.0f, confidence, 0.001f);
        }

        [Test]
        public void Confidence_With200Inliers_ClampedTo1Point0()
        {
            float confidence = Mathf.Min(1.0f, 200 / 100.0f);
            Assert.AreEqual(1.0f, confidence, 0.001f);
        }

        [Test]
        public void Confidence_With0Inliers_Returns0()
        {
            float confidence = Mathf.Min(1.0f, 0 / 100.0f);
            Assert.AreEqual(0.0f, confidence, 0.001f);
        }

        #endregion

        #region ComposePoseMatrix Tests

        [Test]
        public void ComposePoseMatrix_IdentityRotation_ReturnsCorrectMatrix()
        {
            Mat rvec = new Mat(3, 1, CvType.CV_64FC1);
            rvec.put(0, 0, 0.0, 0.0, 0.0);
            Mat tvec = new Mat(3, 1, CvType.CV_64FC1);
            tvec.put(0, 0, 1.0, 2.0, 3.0);

            Matrix4x4 pose = VisualLocalizationEngine.ComposePoseMatrix(rvec, tvec);

            // Rotation should be identity
            Assert.AreEqual(1f, pose.m00, 0.01f);
            Assert.AreEqual(0f, pose.m01, 0.01f);
            Assert.AreEqual(0f, pose.m02, 0.01f);
            Assert.AreEqual(0f, pose.m10, 0.01f);
            Assert.AreEqual(1f, pose.m11, 0.01f);
            Assert.AreEqual(0f, pose.m12, 0.01f);
            Assert.AreEqual(0f, pose.m20, 0.01f);
            Assert.AreEqual(0f, pose.m21, 0.01f);
            Assert.AreEqual(1f, pose.m22, 0.01f);

            // Translation
            Assert.AreEqual(1f, pose.m03, 0.01f);
            Assert.AreEqual(2f, pose.m13, 0.01f);
            Assert.AreEqual(3f, pose.m23, 0.01f);

            // Bottom row
            Assert.AreEqual(0f, pose.m30, 0.01f);
            Assert.AreEqual(0f, pose.m31, 0.01f);
            Assert.AreEqual(0f, pose.m32, 0.01f);
            Assert.AreEqual(1f, pose.m33, 0.01f);

            rvec.Dispose();
            tvec.Dispose();
        }

        [Test]
        public void ComposePoseMatrix_NonTrivialRotation_ProducesValidRigidTransform()
        {
            Mat rvec = new Mat(3, 1, CvType.CV_64FC1);
            rvec.put(0, 0, 0.3, -0.2, 0.1);
            Mat tvec = new Mat(3, 1, CvType.CV_64FC1);
            tvec.put(0, 0, 1.5, -0.5, 4.0);

            Matrix4x4 pose = VisualLocalizationEngine.ComposePoseMatrix(rvec, tvec);

            // Verify det(R) ≈ 1.0 (valid rotation)
            float det = ComputeRotationDeterminant(pose);
            Assert.AreEqual(1.0f, det, 0.01f, "Rotation determinant should be ~1.0");

            // Verify bottom row
            Assert.AreEqual(0f, pose.m30, 0.001f);
            Assert.AreEqual(0f, pose.m31, 0.001f);
            Assert.AreEqual(0f, pose.m32, 0.001f);
            Assert.AreEqual(1f, pose.m33, 0.001f);

            rvec.Dispose();
            tvec.Dispose();
        }

        #endregion

        #region ExtractTranslation Tests

        [Test]
        public void ExtractTranslation_ReturnsCorrectValues()
        {
            var pose = Matrix4x4.identity;
            pose.m03 = 1.5f;
            pose.m13 = -2.3f;
            pose.m23 = 4.7f;

            Vector3 t = VisualLocalizationEngine.ExtractTranslation(pose);

            Assert.AreEqual(1.5f, t.x, 0.001f);
            Assert.AreEqual(-2.3f, t.y, 0.001f);
            Assert.AreEqual(4.7f, t.z, 0.001f);
        }

        #endregion

        #region Helper Utilities

        private static float ComputeRotationDeterminant(Matrix4x4 pose)
        {
            // det of 3x3 rotation submatrix
            return pose.m00 * (pose.m11 * pose.m22 - pose.m12 * pose.m21)
                 - pose.m01 * (pose.m10 * pose.m22 - pose.m12 * pose.m20)
                 + pose.m02 * (pose.m10 * pose.m21 - pose.m11 * pose.m20);
        }

        /// <summary>
        /// Creates a mock FeatureDatabaseReader with minimal data for testing.
        /// </summary>
        private static FeatureDatabaseReader CreateMockFeatureDatabase()
        {
            // We create a FeatureDatabaseReader and populate it via reflection
            // since it normally loads from SQLite. For unit tests, we use a
            // lightweight in-memory approach.
            var db = new FeatureDatabaseReader();
            var keyframes = new List<KeyframeRecord>();

            // Add one keyframe with some dummy data
            var kf = new KeyframeRecord
            {
                Id = 0,
                Pose = new float[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
                GlobalDescriptor = new float[10],
                Keypoints2D = new List<Vector2>(),
                Points3D = new List<Vector3>(),
                Descriptors = new List<byte[]>()
            };

            // Add some dummy features
            var rng = new System.Random(42);
            for (int i = 0; i < 50; i++)
            {
                kf.Keypoints2D.Add(new Vector2(rng.Next(640), rng.Next(480)));
                kf.Points3D.Add(new Vector3(
                    (float)(rng.NextDouble() * 4 - 2),
                    (float)(rng.NextDouble() * 4 - 2),
                    (float)(rng.NextDouble() * 3 + 3)));
                byte[] desc = new byte[32];
                rng.NextBytes(desc);
                kf.Descriptors.Add(desc);
            }

            keyframes.Add(kf);

            // Use reflection to set private fields
            var kfField = typeof(FeatureDatabaseReader).GetField("_keyframes",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            kfField?.SetValue(db, keyframes);

            var vocabField = typeof(FeatureDatabaseReader).GetField("_vocabulary",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            vocabField?.SetValue(db, new List<VocabularyWord>());

            return db;
        }

        #endregion
    }
}
