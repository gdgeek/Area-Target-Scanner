"""Tests for processing_pipeline.feature_db — SQLite round-trip."""

from __future__ import annotations

import math
import os
import sqlite3
import tempfile

import numpy as np
import pytest
from sklearn.cluster import KMeans

from processing_pipeline.feature_db import (
    load_feature_database,
    save_feature_database,
)
from processing_pipeline.models import FeatureDatabase, KeyframeData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keyframe(image_id: int, n_features: int = 50, seed: int = 0) -> KeyframeData:
    """Create a synthetic keyframe with random data."""
    rng = np.random.RandomState(seed + image_id)
    kps = [(rng.uniform(0, 640), rng.uniform(0, 480)) for _ in range(n_features)]
    pts3d = [
        (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 3))
        for _ in range(n_features)
    ]
    descriptors = rng.randint(0, 256, size=(n_features, 32), dtype=np.uint8)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = rng.uniform(-2, 2, size=3)
    return KeyframeData(
        image_id=image_id,
        keypoints=kps,
        descriptors=descriptors,
        points_3d=pts3d,
        camera_pose=pose,
    )


def _make_feature_database(n_keyframes: int = 3, n_features: int = 50) -> FeatureDatabase:
    """Build a complete FeatureDatabase with vocabulary and BoW vectors."""
    from processing_pipeline.feature_extraction import _hamming_word_assignment

    keyframes = [_make_keyframe(i, n_features) for i in range(n_keyframes)]

    # Build vocabulary via KMeans, then extract uint8 medoids
    all_desc_uint8 = np.vstack([kf.descriptors for kf in keyframes])
    all_desc_float = all_desc_uint8.astype(np.float64)
    k = min(10, len(all_desc_float))  # small K for speed
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=2, max_iter=20)
    kmeans.fit(all_desc_float)

    # Compute medoids (real uint8 descriptors closest to each center)
    labels = kmeans.labels_
    centers_uint8 = kmeans.cluster_centers_.astype(np.uint8)
    medoids = np.zeros((k, all_desc_uint8.shape[1]), dtype=np.uint8)
    for c in range(k):
        mask = labels == c
        cluster_descs = all_desc_uint8[mask]
        if len(cluster_descs) == 0:
            medoids[c] = centers_uint8[c]
            continue
        xor = np.bitwise_xor(cluster_descs, centers_uint8[c])
        hamming_dists = np.zeros(len(cluster_descs), dtype=np.int64)
        for byte_col in range(xor.shape[1]):
            hamming_dists += np.array(
                [bin(b).count("1") for b in xor[:, byte_col]], dtype=np.int64
            )
        medoids[c] = cluster_descs[np.argmin(hamming_dists)]

    # Compute BoW vectors using Hamming distance word assignment
    bow = np.zeros((n_keyframes, k), dtype=np.float64)
    for i, kf in enumerate(keyframes):
        word_labels = _hamming_word_assignment(kf.descriptors, medoids)
        for lbl in word_labels:
            bow[i, lbl] += 1.0
        l1 = bow[i].sum()
        if l1 > 0:
            bow[i] /= l1

    return FeatureDatabase(
        keyframes=keyframes,
        global_descriptors=bow,
        vocabulary=medoids,
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSQLiteSchema:
    """Verify the SQLite database has the correct table structure."""

    def test_tables_exist(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        db = _make_feature_database()
        save_feature_database(db, db_path)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = sorted(row[0] for row in cur.fetchall())
        conn.close()

        assert "features" in tables
        assert "keyframes" in tables
        assert "vocabulary" in tables

    def test_keyframes_columns(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        save_feature_database(_make_feature_database(), db_path)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(keyframes)")
        cols = {row[1] for row in cur.fetchall()}
        conn.close()

        assert cols == {"id", "pose", "global_descriptor"}

    def test_features_columns(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        save_feature_database(_make_feature_database(), db_path)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(features)")
        cols = {row[1] for row in cur.fetchall()}
        conn.close()

        assert cols == {"id", "keyframe_id", "x", "y", "x3d", "y3d", "z3d", "descriptor"}

    def test_vocabulary_columns(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        save_feature_database(_make_feature_database(), db_path)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(vocabulary)")
        cols = {row[1] for row in cur.fetchall()}
        conn.close()

        assert cols == {"word_id", "descriptor", "idf_weight"}


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify save → load preserves all data."""

    def test_keyframe_count_preserved(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        original = _make_feature_database(n_keyframes=4)
        save_feature_database(original, db_path)
        loaded = load_feature_database(db_path)

        assert len(loaded.keyframes) == len(original.keyframes)

    def test_keyframe_ids_preserved(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        original = _make_feature_database()
        save_feature_database(original, db_path)
        loaded = load_feature_database(db_path)

        orig_ids = [kf.image_id for kf in original.keyframes]
        loaded_ids = [kf.image_id for kf in loaded.keyframes]
        assert loaded_ids == orig_ids

    def test_pose_preserved(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        original = _make_feature_database()
        save_feature_database(original, db_path)
        loaded = load_feature_database(db_path)

        for orig_kf, loaded_kf in zip(original.keyframes, loaded.keyframes):
            np.testing.assert_array_almost_equal(loaded_kf.camera_pose, orig_kf.camera_pose)

    def test_keypoints_preserved(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        original = _make_feature_database()
        save_feature_database(original, db_path)
        loaded = load_feature_database(db_path)

        for orig_kf, loaded_kf in zip(original.keyframes, loaded.keyframes):
            assert len(loaded_kf.keypoints) == len(orig_kf.keypoints)
            for (ox, oy), (lx, ly) in zip(orig_kf.keypoints, loaded_kf.keypoints):
                assert abs(lx - ox) < 1e-6
                assert abs(ly - oy) < 1e-6

    def test_points_3d_preserved(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        original = _make_feature_database()
        save_feature_database(original, db_path)
        loaded = load_feature_database(db_path)

        for orig_kf, loaded_kf in zip(original.keyframes, loaded.keyframes):
            assert len(loaded_kf.points_3d) == len(orig_kf.points_3d)
            for op, lp in zip(orig_kf.points_3d, loaded_kf.points_3d):
                for a, b in zip(op, lp):
                    assert abs(a - b) < 1e-6

    def test_descriptors_preserved(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        original = _make_feature_database()
        save_feature_database(original, db_path)
        loaded = load_feature_database(db_path)

        for orig_kf, loaded_kf in zip(original.keyframes, loaded.keyframes):
            np.testing.assert_array_equal(loaded_kf.descriptors, orig_kf.descriptors)

    def test_global_descriptors_preserved(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        original = _make_feature_database()
        save_feature_database(original, db_path)
        loaded = load_feature_database(db_path)

        assert loaded.global_descriptors is not None
        np.testing.assert_array_almost_equal(
            loaded.global_descriptors, original.global_descriptors
        )

    def test_vocabulary_centres_preserved(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        original = _make_feature_database()
        save_feature_database(original, db_path)
        loaded = load_feature_database(db_path)

        assert loaded.vocabulary is not None
        np.testing.assert_array_equal(
            loaded.vocabulary,
            original.vocabulary,
        )

    def test_vocabulary_word_assignment_consistent(self, tmp_path):
        """Loaded vocabulary should assign the same words as the original via Hamming distance."""
        from processing_pipeline.feature_extraction import _hamming_word_assignment

        db_path = str(tmp_path / "features.db")
        original = _make_feature_database()
        save_feature_database(original, db_path)
        loaded = load_feature_database(db_path)

        test_desc = original.keyframes[0].descriptors
        orig_labels = _hamming_word_assignment(test_desc, original.vocabulary)
        loaded_labels = _hamming_word_assignment(test_desc, loaded.vocabulary)
        np.testing.assert_array_equal(loaded_labels, orig_labels)


# ---------------------------------------------------------------------------
# IDF weight tests
# ---------------------------------------------------------------------------


class TestIDFWeights:
    """Verify IDF weights in the vocabulary table."""

    def test_idf_weights_stored(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        db = _make_feature_database()
        save_feature_database(db, db_path)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT idf_weight FROM vocabulary")
        weights = [row[0] for row in cur.fetchall()]
        conn.close()

        assert len(weights) == len(db.vocabulary)
        # All weights should be finite (IDF can be negative when df >= N)
        for w in weights:
            assert math.isfinite(w)

    def test_idf_formula(self, tmp_path):
        """IDF should equal log(N / (1 + df)) for each word."""
        from processing_pipeline.feature_extraction import _hamming_word_assignment

        db_path = str(tmp_path / "features.db")
        db = _make_feature_database(n_keyframes=5, n_features=80)
        save_feature_database(db, db_path)

        n_kf = len(db.keyframes)
        medoids = db.vocabulary

        # Manually compute expected doc frequencies using Hamming distance
        doc_freq = np.zeros(len(medoids), dtype=np.float64)
        for kf in db.keyframes:
            labels = _hamming_word_assignment(kf.descriptors, medoids)
            for w in set(labels.tolist()):
                doc_freq[w] += 1.0

        expected_idf = [math.log(n_kf / (1.0 + doc_freq[w])) for w in range(len(medoids))]

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT word_id, idf_weight FROM vocabulary ORDER BY word_id")
        rows = cur.fetchall()
        conn.close()

        for word_id, idf_weight in rows:
            assert abs(idf_weight - expected_idf[word_id]) < 1e-10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for save/load."""

    def test_empty_database(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        db = FeatureDatabase(keyframes=[], global_descriptors=None, vocabulary=None)
        save_feature_database(db, db_path)
        loaded = load_feature_database(db_path)

        assert len(loaded.keyframes) == 0
        assert loaded.global_descriptors is None
        assert loaded.vocabulary is None

    def test_no_vocabulary(self, tmp_path):
        """Database with keyframes but no vocabulary."""
        db_path = str(tmp_path / "features.db")
        kf = _make_keyframe(0)
        db = FeatureDatabase(keyframes=[kf], global_descriptors=None, vocabulary=None)
        save_feature_database(db, db_path)
        loaded = load_feature_database(db_path)

        assert len(loaded.keyframes) == 1
        assert loaded.vocabulary is None

    def test_single_keyframe(self, tmp_path):
        db_path = str(tmp_path / "features.db")
        db = _make_feature_database(n_keyframes=1, n_features=30)
        save_feature_database(db, db_path)
        loaded = load_feature_database(db_path)

        assert len(loaded.keyframes) == 1
        np.testing.assert_array_equal(
            loaded.keyframes[0].descriptors, db.keyframes[0].descriptors
        )
