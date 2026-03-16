using System;
using System.Collections.Generic;
using UnityEngine;
using OpenCvForUnity.CoreModule;
using OpenCvForUnity.Features2dModule;
using OpenCvForUnity.Calib3dModule;

namespace AreaTargetPlugin
{
    /// <summary>
    /// Visual localization engine that performs ORB feature extraction, BoW-based
    /// keyframe retrieval, feature matching, and PnP pose estimation.
    /// Implements the core Algorithm 4 from the design document.
    /// </summary>
    public class VisualLocalizationEngine : IDisposable
    {
        // ORB parameters
        private const int OrbNFeatures = 1000;
        private const int MinFeatureCount = 10;

        // Matching parameters
        private const float LoweRatioThreshold = 0.75f;
        private const int MinGoodMatches = 20;

        // PnP RANSAC parameters
        private const int PnpIterations = 100;
        private const float PnpReprojError = 8.0f;
        private const double PnpConfidence = 0.99;

        // Inlier thresholds
        private const int MinInlierCount = 20;
        private const float MaxConfidenceDivisor = 100.0f;

        // Nearby keyframe search parameters
        private const float NearbyRadius = 5.0f;
        private const int MaxNearbyKeyframes = 5;
        private const int GlobalTopK = 10;

        private ORB _orb;
        private DescriptorMatcher _matcher;
        private FeatureDatabaseReader _featureDb;
        private bool _disposed;

        /// <summary>
        /// The last successfully computed pose (null if no valid pose yet).
        /// </summary>
        public Matrix4x4? LastValidPose { get; private set; }

        /// <summary>
        /// Initializes the localization engine with the given feature database.
        /// </summary>
        public bool Initialize(FeatureDatabaseReader featureDb)
        {
            if (featureDb == null || featureDb.KeyframeCount == 0)
            {
                Debug.LogError("[VisualLocalizationEngine] Feature database is null or empty.");
                return false;
            }

            _featureDb = featureDb;
            _orb = ORB.create(OrbNFeatures);
            _matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            LastValidPose = null;
            return true;
        }

        /// <summary>
        /// Processes a camera frame and returns a tracking result.
        /// Implements: ORB extraction → BoW retrieval → feature matching → PnP solve.
        /// </summary>
        public TrackingResult ProcessFrame(CameraFrame frame)
        {
            if (_disposed || _featureDb == null)
            {
                return CreateLostResult();
            }

            // Step 1: Extract ORB features from the current frame
            Mat imageMat = new Mat(frame.Height, frame.Width, CvType.CV_8UC1);
            imageMat.put(0, 0, frame.ImageData);

            MatOfKeyPoint queryKeypoints = new MatOfKeyPoint();
            Mat queryDescriptors = new Mat();
            _orb.detectAndCompute(imageMat, new Mat(), queryKeypoints, queryDescriptors);

            int featureCount = queryKeypoints.toArray().Length;
            imageMat.Dispose();

            // Requirement 11.8: If < 10 features, return LOST
            if (featureCount < MinFeatureCount || queryDescriptors.empty())
            {
                queryKeypoints.Dispose();
                queryDescriptors.Dispose();
                return CreateLostResult();
            }

            // Step 2: Compute BoW vector for coarse localization
            float[] queryBoW = ComputeBoWVector(queryDescriptors);

            // Step 3: Find candidate keyframes
            List<KeyframeRecord> candidates;
            if (LastValidPose.HasValue)
            {
                // Requirement 11.3: Search nearby keyframes (radius 5.0m, max 5)
                Vector3 lastPos = ExtractTranslation(LastValidPose.Value);
                candidates = _featureDb.GetNearbyKeyframes(lastPos, NearbyRadius, MaxNearbyKeyframes);
            }
            else
            {
                // Requirement 11.4: Global BoW similarity search (top-10)
                candidates = _featureDb.GetTopKByBoWSimilarity(queryBoW, GlobalTopK);
            }

            // Step 4: Feature matching + PnP for each candidate
            TrackingResult bestResult = CreateLostResult();
            int bestInlierCount = 0;

            KeyPoint[] queryKpArray = queryKeypoints.toArray();

            foreach (var keyframe in candidates)
            {
                var matchResult = MatchAndSolvePnP(
                    queryKpArray, queryDescriptors, keyframe, frame.Intrinsics);

                if (matchResult.HasValue && matchResult.Value.inlierCount > bestInlierCount)
                {
                    bestInlierCount = matchResult.Value.inlierCount;
                    bestResult = matchResult.Value.result;
                }
            }

            queryKeypoints.Dispose();
            queryDescriptors.Dispose();

            // Update last valid pose if tracking
            if (bestResult.State == TrackingState.TRACKING)
            {
                LastValidPose = bestResult.Pose;
            }

            return bestResult;
        }

        private struct PnPMatchResult
        {
            public TrackingResult result;
            public int inlierCount;
        }

        /// <summary>
        /// Matches query descriptors against a keyframe and solves PnP if enough matches.
        /// </summary>
        private PnPMatchResult? MatchAndSolvePnP(
            KeyPoint[] queryKpArray,
            Mat queryDescriptors,
            KeyframeRecord keyframe,
            Matrix4x4 intrinsics)
        {
            if (keyframe.Descriptors.Count == 0) return null;

            // Build keyframe descriptor Mat
            Mat kfDescriptors = new Mat(keyframe.Descriptors.Count, 32, CvType.CV_8UC1);
            for (int i = 0; i < keyframe.Descriptors.Count; i++)
            {
                kfDescriptors.put(i, 0, keyframe.Descriptors[i]);
            }

            // KNN match (k=2) for Lowe's ratio test
            List<MatOfDMatch> knnMatches = new List<MatOfDMatch>();
            _matcher.knnMatch(queryDescriptors, kfDescriptors, knnMatches, 2);

            // Requirement 11.5: Lowe ratio test (threshold = 0.75)
            List<DMatch> goodMatches = new List<DMatch>();
            foreach (var matchPair in knnMatches)
            {
                DMatch[] matches = matchPair.toArray();
                if (matches.Length >= 2 && matches[0].distance < LoweRatioThreshold * matches[1].distance)
                {
                    goodMatches.Add(matches[0]);
                }
                matchPair.Dispose();
            }

            kfDescriptors.Dispose();

            // Requirement 11.6: Need >= 20 good matches for PnP
            if (goodMatches.Count < MinGoodMatches) return null;

            // Build 2D-3D correspondences
            MatOfPoint2f points2D = new MatOfPoint2f();
            MatOfPoint3f points3D = new MatOfPoint3f();

            List<Point> pts2d = new List<Point>();
            List<Point3> pts3d = new List<Point3>();

            foreach (var m in goodMatches)
            {
                var kp = queryKpArray[m.queryIdx];
                var pt3d = keyframe.Points3D[m.trainIdx];
                pts2d.Add(new Point(kp.pt.x, kp.pt.y));
                pts3d.Add(new Point3(pt3d.x, pt3d.y, pt3d.z));
            }

            points2D.fromList(pts2d);
            points3D.fromList(pts3d);

            // Build camera matrix from intrinsics
            Mat cameraMat = new Mat(3, 3, CvType.CV_64FC1);
            cameraMat.put(0, 0,
                intrinsics.m00, intrinsics.m01, intrinsics.m02,
                intrinsics.m10, intrinsics.m11, intrinsics.m12,
                intrinsics.m20, intrinsics.m21, intrinsics.m22);

            Mat distCoeffs = new Mat();
            Mat rvec = new Mat();
            Mat tvec = new Mat();
            Mat inliersMat = new Mat();

            // Requirement 11.6: PnP+RANSAC (iterations=100, reprojError=8.0, confidence=0.99)
            bool success = Calib3d.solvePnPRansac(
                points3D, points2D, cameraMat, distCoeffs,
                rvec, tvec, false,
                PnpIterations, PnpReprojError, PnpConfidence, inliersMat);

            int inlierCount = success ? inliersMat.rows() : 0;

            PnPMatchResult? result = null;

            // Requirement 11.7: PnP success and inliers >= 20 → TRACKING
            if (success && inlierCount >= MinInlierCount)
            {
                Matrix4x4 pose = ComposePoseMatrix(rvec, tvec);
                // Requirement 12.4: confidence = min(1.0, inliers / 100.0)
                float confidence = Mathf.Min(1.0f, inlierCount / MaxConfidenceDivisor);

                result = new PnPMatchResult
                {
                    result = new TrackingResult
                    {
                        State = TrackingState.TRACKING,
                        Pose = pose,
                        Confidence = confidence,
                        MatchedFeatures = inlierCount
                    },
                    inlierCount = inlierCount
                };
            }

            // Cleanup
            points2D.Dispose();
            points3D.Dispose();
            cameraMat.Dispose();
            distCoeffs.Dispose();
            rvec.Dispose();
            tvec.Dispose();
            inliersMat.Dispose();

            return result;
        }

        /// <summary>
        /// Computes a BoW vector from ORB descriptors using the loaded vocabulary.
        /// Each descriptor is assigned to the nearest visual word, and the resulting
        /// histogram is L1-normalized.
        /// </summary>
        internal float[] ComputeBoWVector(Mat descriptors)
        {
            if (_featureDb == null || _featureDb.Vocabulary == null || _featureDb.Vocabulary.Count == 0)
            {
                return new float[0];
            }

            int vocabSize = _featureDb.Vocabulary.Count;
            float[] bow = new float[vocabSize];

            int rows = descriptors.rows();
            int cols = descriptors.cols();
            byte[] descData = new byte[cols];

            for (int i = 0; i < rows; i++)
            {
                descriptors.get(i, 0, descData);

                // Find nearest visual word (Hamming-like distance on float vocabulary)
                int bestWord = 0;
                float bestDist = float.MaxValue;

                for (int w = 0; w < vocabSize; w++)
                {
                    float[] wordDesc = _featureDb.Vocabulary[w].Descriptor;
                    float dist = 0f;
                    int len = Math.Min(cols, wordDesc.Length);
                    for (int d = 0; d < len; d++)
                    {
                        float diff = descData[d] - wordDesc[d];
                        dist += diff * diff;
                    }
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestWord = w;
                    }
                }

                bow[bestWord] += _featureDb.Vocabulary[bestWord].IdfWeight;
            }

            // L1 normalize
            float l1Norm = 0f;
            for (int i = 0; i < vocabSize; i++) l1Norm += Math.Abs(bow[i]);
            if (l1Norm > 1e-9f)
            {
                for (int i = 0; i < vocabSize; i++) bow[i] /= l1Norm;
            }

            return bow;
        }

        /// <summary>
        /// Converts rotation vector and translation vector from PnP into a 4x4 pose matrix.
        /// </summary>
        internal static Matrix4x4 ComposePoseMatrix(Mat rvec, Mat tvec)
        {
            Mat rotMat = new Mat();
            Calib3d.Rodrigues(rvec, rotMat);

            double[] r = new double[9];
            rotMat.get(0, 0, r);

            double[] t = new double[3];
            tvec.get(0, 0, t);

            rotMat.Dispose();

            Matrix4x4 pose = new Matrix4x4();
            // Row-major layout
            pose.m00 = (float)r[0]; pose.m01 = (float)r[1]; pose.m02 = (float)r[2]; pose.m03 = (float)t[0];
            pose.m10 = (float)r[3]; pose.m11 = (float)r[4]; pose.m12 = (float)r[5]; pose.m13 = (float)t[1];
            pose.m20 = (float)r[6]; pose.m21 = (float)r[7]; pose.m22 = (float)r[8]; pose.m23 = (float)t[2];
            pose.m30 = 0f;          pose.m31 = 0f;          pose.m32 = 0f;          pose.m33 = 1f;

            return pose;
        }

        /// <summary>
        /// Extracts the translation component from a 4x4 pose matrix.
        /// </summary>
        internal static Vector3 ExtractTranslation(Matrix4x4 pose)
        {
            return new Vector3(pose.m03, pose.m13, pose.m23);
        }

        private static TrackingResult CreateLostResult()
        {
            return new TrackingResult
            {
                State = TrackingState.LOST,
                Pose = Matrix4x4.identity,
                Confidence = 0f,
                MatchedFeatures = 0
            };
        }

        /// <summary>
        /// Resets the localization engine state (clears last valid pose).
        /// </summary>
        public void ResetState()
        {
            LastValidPose = null;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _orb?.Dispose();
            _matcher?.Dispose();
            _orb = null;
            _matcher = null;
            _featureDb = null;
            LastValidPose = null;
        }
    }
}
