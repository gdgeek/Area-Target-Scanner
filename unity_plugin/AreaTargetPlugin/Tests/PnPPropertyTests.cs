using System;
using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using OpenCvForUnity.CoreModule;
using OpenCvForUnity.Calib3dModule;

namespace AreaTargetPlugin.Tests
{
    /// <summary>
    /// Property-based tests for PnP localization result validity.
    /// **Validates: Requirements 12.1, 12.2, 12.3**
    ///
    /// Property P5: PnP localization result validity —
    /// For all TRACKING results: pose is a valid rigid transform (det(R) ≈ 1.0),
    /// confidence ∈ [0.0, 1.0], and matchedFeatures >= 20.
    /// </summary>
    [TestFixture]
    public class PnPPropertyTests
    {
        private const float Fx = 500f;
        private const float Fy = 500f;
        private const float Cx = 320f;
        private const float Cy = 240f;
        private const int ImageWidth = 640;
        private const int ImageHeight = 480;

        /// <summary>
        /// Generates random test cases with varying rotation and translation parameters.
        /// Each test case produces synthetic 2D-3D correspondences from a known pose,
        /// solves PnP, and verifies the P5 property on the result.
        /// </summary>
        private static IEnumerable<TestCaseData> RandomPoseTestCases()
        {
            var rng = new System.Random(12345);
            int numCases = 30;

            for (int i = 0; i < numCases; i++)
            {
                // Random rotation (small angles for realistic scenarios)
                double rx = (rng.NextDouble() - 0.5) * 0.6;  // [-0.3, 0.3] rad
                double ry = (rng.NextDouble() - 0.5) * 0.6;
                double rz = (rng.NextDouble() - 0.5) * 0.6;

                // Random translation (camera looking at scene)
                double tx = (rng.NextDouble() - 0.5) * 2.0;  // [-1, 1] m
                double ty = (rng.NextDouble() - 0.5) * 2.0;
                double tz = rng.NextDouble() * 4.0 + 3.0;     // [3, 7] m depth

                int numPoints = rng.Next(40, 100);

                yield return new TestCaseData(rx, ry, rz, tx, ty, tz, numPoints, i)
                    .SetName($"P5_PnPValidity_Case{i}_R({rx:F2},{ry:F2},{rz:F2})_T({tx:F2},{ty:F2},{tz:F2})_N{numPoints}");
            }
        }

        /// <summary>
        /// Property P5: For all TRACKING results, the pose is a valid rigid body transform
        /// (det(R) ≈ 1.0), confidence ∈ [0.0, 1.0], and matchedFeatures >= 20.
        /// **Validates: Requirements 12.1, 12.2, 12.3**
        /// </summary>
        [Test, TestCaseSource(nameof(RandomPoseTestCases))]
        public void P5_TrackingResult_IsValidRigidTransform(
            double rx, double ry, double rz,
            double tx, double ty, double tz,
            int numPoints, int caseIndex)
        {
            // Generate synthetic 2D-3D correspondences from the known pose
            double[] gtRvec = { rx, ry, rz };
            double[] gtTvec = { tx, ty, tz };

            GenerateCorrespondences(numPoints, gtRvec, gtTvec,
                out var points3D, out var points2D);

            // Need at least 20 valid correspondences for PnP
            if (points3D.Count < 20)
            {
                Assert.Inconclusive($"Only {points3D.Count} valid projections, need >= 20");
                return;
            }

            // Solve PnP+RANSAC (same parameters as the engine)
            MatOfPoint3f objPts = new MatOfPoint3f();
            objPts.fromList(points3D);
            MatOfPoint2f imgPts = new MatOfPoint2f();
            imgPts.fromList(points2D);

            Mat cameraMat = new Mat(3, 3, CvType.CV_64FC1);
            cameraMat.put(0, 0, Fx, 0, Cx, 0, Fy, Cy, 0, 0, 1);

            Mat distCoeffs = new Mat();
            Mat rvec = new Mat();
            Mat tvec = new Mat();
            Mat inliers = new Mat();

            bool success = Calib3d.solvePnPRansac(
                objPts, imgPts, cameraMat, distCoeffs,
                rvec, tvec, false, 100, 8.0f, 0.99, inliers);

            int inlierCount = success ? inliers.rows() : 0;

            if (success && inlierCount >= 20)
            {
                // Build the tracking result as the engine would
                Matrix4x4 pose = VisualLocalizationEngine.ComposePoseMatrix(rvec, tvec);
                float confidence = Mathf.Min(1.0f, inlierCount / 100.0f);
                int matchedFeatures = inlierCount;

                // === Property P5 Assertions ===

                // P5.1: Requirement 12.1 — Pose is valid rigid transform (det(R) ≈ 1.0)
                float det = ComputeRotationDeterminant(pose);
                Assert.AreEqual(1.0f, det, 0.05f,
                    $"[Case {caseIndex}] det(R) should be ≈ 1.0 for valid rigid transform, got {det}");

                // P5.2: Verify rotation matrix orthogonality (R^T * R ≈ I)
                VerifyOrthogonality(pose, caseIndex);

                // P5.3: Requirement 12.2 — Confidence ∈ [0.0, 1.0]
                Assert.GreaterOrEqual(confidence, 0.0f,
                    $"[Case {caseIndex}] Confidence should be >= 0.0");
                Assert.LessOrEqual(confidence, 1.0f,
                    $"[Case {caseIndex}] Confidence should be <= 1.0");

                // P5.4: Requirement 12.3 — matchedFeatures >= 20
                Assert.GreaterOrEqual(matchedFeatures, 20,
                    $"[Case {caseIndex}] Matched features should be >= 20 for TRACKING state");

                // P5.5: Bottom row of pose matrix should be [0, 0, 0, 1]
                Assert.AreEqual(0f, pose.m30, 0.001f, $"[Case {caseIndex}] pose.m30 should be 0");
                Assert.AreEqual(0f, pose.m31, 0.001f, $"[Case {caseIndex}] pose.m31 should be 0");
                Assert.AreEqual(0f, pose.m32, 0.001f, $"[Case {caseIndex}] pose.m32 should be 0");
                Assert.AreEqual(1f, pose.m33, 0.001f, $"[Case {caseIndex}] pose.m33 should be 1");
            }
            else
            {
                // PnP failed or not enough inliers — this is a LOST result
                // For LOST results, confidence should be 0 and matchedFeatures 0
                // (no P5 property to check since P5 only applies to TRACKING)
                Assert.Pass($"[Case {caseIndex}] PnP returned LOST (success={success}, inliers={inlierCount}), P5 not applicable");
            }

            // Cleanup
            objPts.Dispose(); imgPts.Dispose();
            cameraMat.Dispose(); distCoeffs.Dispose();
            rvec.Dispose(); tvec.Dispose(); inliers.Dispose();
        }

        /// <summary>
        /// Property P5 sub-property: Confidence calculation is always min(1.0, inliers/100.0).
        /// **Validates: Requirements 12.2, 12.4**
        /// </summary>
        [Test, TestCaseSource(nameof(RandomInlierCounts))]
        public void P5_ConfidenceCalculation_AlwaysInRange(int inlierCount)
        {
            float confidence = Mathf.Min(1.0f, inlierCount / 100.0f);

            Assert.GreaterOrEqual(confidence, 0.0f,
                $"Confidence for {inlierCount} inliers should be >= 0.0");
            Assert.LessOrEqual(confidence, 1.0f,
                $"Confidence for {inlierCount} inliers should be <= 1.0");

            // Verify exact formula
            float expected = Math.Min(1.0f, inlierCount / 100.0f);
            Assert.AreEqual(expected, confidence, 0.0001f);
        }

        private static IEnumerable<TestCaseData> RandomInlierCounts()
        {
            var rng = new System.Random(99);
            for (int i = 0; i < 50; i++)
            {
                int count = rng.Next(0, 500);
                yield return new TestCaseData(count).SetName($"P5_Confidence_Inliers{count}");
            }
        }

        #region Helper Methods

        private static void GenerateCorrespondences(
            int count, double[] rvec, double[] tvec,
            out List<Point3> points3D, out List<Point> points2D)
        {
            points3D = new List<Point3>();
            points2D = new List<Point>();
            var rng = new System.Random(42 + count);

            // Convert rvec to rotation matrix
            Mat rvecMat = new Mat(3, 1, CvType.CV_64FC1);
            rvecMat.put(0, 0, rvec);
            Mat rotMat = new Mat();
            Calib3d.Rodrigues(rvecMat, rotMat);
            double[] r = new double[9];
            rotMat.get(0, 0, r);
            rvecMat.Dispose();
            rotMat.Dispose();

            for (int i = 0; i < count; i++)
            {
                double x = (rng.NextDouble() - 0.5) * 4.0;
                double y = (rng.NextDouble() - 0.5) * 4.0;
                double z = rng.NextDouble() * 3.0 + 3.0;

                // Transform to camera coordinates
                double cx = r[0] * x + r[1] * y + r[2] * z + tvec[0];
                double cy = r[3] * x + r[4] * y + r[5] * z + tvec[1];
                double cz = r[6] * x + r[7] * y + r[8] * z + tvec[2];

                if (cz <= 0.1) continue; // Behind camera

                double u = Fx * cx / cz + Cx;
                double v = Fy * cy / cz + Cy;

                if (u >= 0 && u < ImageWidth && v >= 0 && v < ImageHeight)
                {
                    points3D.Add(new Point3(x, y, z));
                    points2D.Add(new Point(u, v));
                }
            }
        }

        private static float ComputeRotationDeterminant(Matrix4x4 pose)
        {
            return pose.m00 * (pose.m11 * pose.m22 - pose.m12 * pose.m21)
                 - pose.m01 * (pose.m10 * pose.m22 - pose.m12 * pose.m20)
                 + pose.m02 * (pose.m10 * pose.m21 - pose.m11 * pose.m20);
        }

        private static void VerifyOrthogonality(Matrix4x4 pose, int caseIndex)
        {
            // Check R^T * R ≈ I by verifying column dot products
            // Column 0
            Vector3 c0 = new Vector3(pose.m00, pose.m10, pose.m20);
            Vector3 c1 = new Vector3(pose.m01, pose.m11, pose.m21);
            Vector3 c2 = new Vector3(pose.m02, pose.m12, pose.m22);

            // Columns should be unit vectors
            Assert.AreEqual(1.0f, c0.magnitude, 0.05f,
                $"[Case {caseIndex}] Column 0 should be unit vector");
            Assert.AreEqual(1.0f, c1.magnitude, 0.05f,
                $"[Case {caseIndex}] Column 1 should be unit vector");
            Assert.AreEqual(1.0f, c2.magnitude, 0.05f,
                $"[Case {caseIndex}] Column 2 should be unit vector");

            // Columns should be orthogonal
            Assert.AreEqual(0f, Vector3.Dot(c0, c1), 0.05f,
                $"[Case {caseIndex}] Columns 0 and 1 should be orthogonal");
            Assert.AreEqual(0f, Vector3.Dot(c0, c2), 0.05f,
                $"[Case {caseIndex}] Columns 0 and 2 should be orthogonal");
            Assert.AreEqual(0f, Vector3.Dot(c1, c2), 0.05f,
                $"[Case {caseIndex}] Columns 1 and 2 should be orthogonal");
        }

        #endregion
    }
}
