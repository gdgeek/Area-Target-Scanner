using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Microsoft.Data.Sqlite;

namespace AreaTargetPlugin
{
    /// <summary>
    /// Represents a single keyframe loaded from the feature database.
    /// </summary>
    public class KeyframeRecord
    {
        public int Id;
        public float[] Pose;           // 4x4 matrix as 16 floats (row-major)
        public float[] GlobalDescriptor; // BoW vector
        public List<Vector2> Keypoints2D;
        public List<Vector3> Points3D;
        public List<byte[]> Descriptors; // Each is 32-byte ORB descriptor
    }

    /// <summary>
    /// Represents a visual word from the vocabulary table.
    /// </summary>
    public class VocabularyWord
    {
        public int WordId;
        public byte[] Descriptor;
        public float IdfWeight;
    }

    /// <summary>
    /// Reads the SQLite feature database produced by the processing pipeline.
    /// Provides keyframe data, vocabulary, and search methods for visual localization.
    /// </summary>
    public class FeatureDatabaseReader : IDisposable
    {
        private List<KeyframeRecord> _keyframes;
        private List<VocabularyWord> _vocabulary;
        private bool _disposed;

        public IReadOnlyList<KeyframeRecord> Keyframes => _keyframes;
        public IReadOnlyList<VocabularyWord> Vocabulary => _vocabulary;
        public int KeyframeCount => _keyframes?.Count ?? 0;

        /// <summary>
        /// Loads the feature database from the given SQLite file path.
        /// </summary>
        public bool Load(string dbPath)
        {
            if (string.IsNullOrEmpty(dbPath) || !File.Exists(dbPath))
            {
                Debug.LogError($"[FeatureDatabaseReader] Database file not found: {dbPath}");
                return false;
            }

            _keyframes = new List<KeyframeRecord>();
            _vocabulary = new List<VocabularyWord>();

            string connectionString = $"Data Source={dbPath}";
            try
            {
                using (var conn = new SqliteConnection(connectionString))
                {
                    conn.Open();
                    LoadKeyframes(conn);
                    LoadVocabulary(conn);
                }
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[FeatureDatabaseReader] Failed to load database: {ex.Message}");
                return false;
            }
        }

        private void LoadKeyframes(SqliteConnection conn)
        {
            // 1. Load all keyframe records into a dictionary keyed by id
            var kfCmd = conn.CreateCommand();
            kfCmd.CommandText = "SELECT id, pose, global_descriptor FROM keyframes ORDER BY id";
            var kfMap = new Dictionary<int, KeyframeRecord>();
            using (var reader = kfCmd.ExecuteReader())
            {
                while (reader.Read())
                {
                    var kf = new KeyframeRecord
                    {
                        Id = reader.GetInt32(0),
                        Keypoints2D = new List<Vector2>(),
                        Points3D = new List<Vector3>(),
                        Descriptors = new List<byte[]>()
                    };

                    // Parse pose (stored as 16 doubles, 128 bytes)
                    byte[] poseBlob = (byte[])reader["pose"];
                    kf.Pose = new float[16];
                    for (int i = 0; i < 16; i++)
                    {
                        kf.Pose[i] = (float)BitConverter.ToDouble(poseBlob, i * 8);
                    }

                    // Parse global descriptor if present
                    if (!reader.IsDBNull(2))
                    {
                        byte[] gdBlob = (byte[])reader["global_descriptor"];
                        int gdLen = gdBlob.Length / 8;
                        kf.GlobalDescriptor = new float[gdLen];
                        for (int i = 0; i < gdLen; i++)
                        {
                            kf.GlobalDescriptor[i] = (float)BitConverter.ToDouble(gdBlob, i * 8);
                        }
                    }

                    kfMap[kf.Id] = kf;
                    _keyframes.Add(kf);
                }
            }

            // 2. Load all features in a single query, group by keyframe_id in memory
            var featCmd = conn.CreateCommand();
            featCmd.CommandText = "SELECT keyframe_id, x, y, x3d, y3d, z3d, descriptor FROM features ORDER BY keyframe_id, id";
            using (var reader = featCmd.ExecuteReader())
            {
                while (reader.Read())
                {
                    int kfId = reader.GetInt32(0);
                    if (kfMap.TryGetValue(kfId, out var kf))
                    {
                        float x = (float)reader.GetDouble(1);
                        float y = (float)reader.GetDouble(2);
                        float x3d = (float)reader.GetDouble(3);
                        float y3d = (float)reader.GetDouble(4);
                        float z3d = (float)reader.GetDouble(5);
                        byte[] desc = (byte[])reader["descriptor"];

                        kf.Keypoints2D.Add(new Vector2(x, y));
                        kf.Points3D.Add(new Vector3(x3d, y3d, z3d));
                        kf.Descriptors.Add(desc);
                    }
                }
            }
        }

        private void LoadVocabulary(SqliteConnection conn)
        {
            var cmd = conn.CreateCommand();
            cmd.CommandText = "SELECT word_id, descriptor, idf_weight FROM vocabulary ORDER BY word_id";
            using (var reader = cmd.ExecuteReader())
            {
                while (reader.Read())
                {
                    var word = new VocabularyWord
                    {
                        WordId = reader.GetInt32(0),
                        IdfWeight = (float)reader.GetDouble(2)
                    };

                    byte[] descBlob = (byte[])reader["descriptor"];
                    word.Descriptor = descBlob;

                    _vocabulary.Add(word);
                }
            }
        }

        /// <summary>
        /// Returns keyframes within the given radius of a position (extracted from pose translation).
        /// Used when a previous valid pose exists for local search.
        /// </summary>
        public List<KeyframeRecord> GetNearbyKeyframes(Vector3 position, float radius, int maxCount)
        {
            var candidates = new List<(KeyframeRecord kf, float dist)>();
            float radiusSq = radius * radius;

            foreach (var kf in _keyframes)
            {
                // Extract translation from the 4x4 pose matrix (row-major: indices 3, 7, 11)
                Vector3 kfPos = new Vector3(kf.Pose[3], kf.Pose[7], kf.Pose[11]);
                float distSq = (kfPos - position).sqrMagnitude;
                if (distSq <= radiusSq)
                {
                    candidates.Add((kf, distSq));
                }
            }

            candidates.Sort((a, b) => a.dist.CompareTo(b.dist));
            var result = new List<KeyframeRecord>();
            for (int i = 0; i < Math.Min(maxCount, candidates.Count); i++)
            {
                result.Add(candidates[i].kf);
            }
            return result;
        }

        /// <summary>
        /// Returns the top-K keyframes by BoW similarity to the given query BoW vector.
        /// Used for global relocalization when no previous pose is available.
        /// </summary>
        public List<KeyframeRecord> GetTopKByBoWSimilarity(float[] queryBoW, int k)
        {
            var scored = new List<(KeyframeRecord kf, float score)>();

            foreach (var kf in _keyframes)
            {
                if (kf.GlobalDescriptor == null) continue;
                float score = ComputeBoWSimilarity(queryBoW, kf.GlobalDescriptor);
                scored.Add((kf, score));
            }

            scored.Sort((a, b) => b.score.CompareTo(a.score)); // descending
            var result = new List<KeyframeRecord>();
            for (int i = 0; i < Math.Min(k, scored.Count); i++)
            {
                result.Add(scored[i].kf);
            }
            return result;
        }

        /// <summary>
        /// Computes cosine similarity between two BoW vectors.
        /// </summary>
        private static float ComputeBoWSimilarity(float[] a, float[] b)
        {
            if (a == null || b == null) return 0f;
            int len = Math.Min(a.Length, b.Length);
            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < len; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            float denom = (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
            return denom > 1e-9f ? dot / denom : 0f;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _keyframes = null;
            _vocabulary = null;
        }
    }
}
