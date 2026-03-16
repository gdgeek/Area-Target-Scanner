"""SQLite persistence for the visual feature database.

Provides :func:`save_feature_database` and :func:`load_feature_database` to
round-trip a :class:`~processing_pipeline.models.FeatureDatabase` through a
SQLite file with three tables: ``keyframes``, ``features``, and ``vocabulary``.
"""

from __future__ import annotations

import math
import sqlite3
from typing import List

import numpy as np

from processing_pipeline.models import FeatureDatabase, KeyframeData

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS keyframes (
    id INTEGER PRIMARY KEY,
    pose BLOB NOT NULL,
    global_descriptor BLOB
);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY,
    keyframe_id INTEGER NOT NULL REFERENCES keyframes(id),
    x REAL NOT NULL,
    y REAL NOT NULL,
    x3d REAL NOT NULL,
    y3d REAL NOT NULL,
    z3d REAL NOT NULL,
    descriptor BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS vocabulary (
    word_id INTEGER PRIMARY KEY,
    descriptor BLOB NOT NULL,
    idf_weight REAL NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_idf_weights(
    keyframes: List[KeyframeData],
    n_words: int,
    vocabulary_centers: np.ndarray,
) -> np.ndarray:
    """Compute IDF weight for each visual word.

    IDF = log(N / (1 + df)) where *N* is the total number of keyframes and
    *df* is the number of keyframes that contain at least one descriptor
    assigned to that word.

    Args:
        keyframes: Keyframe list from the feature database.
        n_words: Number of visual words (K).
        vocabulary_centers: (K, D) cluster centres used for word assignment.

    Returns:
        (K,) array of IDF weights.
    """
    n_keyframes = len(keyframes)
    doc_freq = np.zeros(n_words, dtype=np.float64)

    for kf in keyframes:
        if kf.descriptors is None or len(kf.descriptors) == 0:
            continue
        desc_float = kf.descriptors.astype(np.float64)
        # Use pairwise distance to assign words (same logic as KMeans.predict)
        # but we avoid requiring a full KMeans object here.
        diffs = desc_float[:, np.newaxis, :] - vocabulary_centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        unique_words = set(labels.tolist())
        for w in unique_words:
            doc_freq[w] += 1.0

    idf = np.array(
        [math.log(n_keyframes / (1.0 + doc_freq[w])) for w in range(n_words)],
        dtype=np.float64,
    )
    return idf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_feature_database(db: FeatureDatabase, db_path: str) -> None:
    """Persist a :class:`FeatureDatabase` to a SQLite file.

    Creates (or overwrites) the database at *db_path* with three tables:
    ``keyframes``, ``features``, and ``vocabulary``.

    Args:
        db: The in-memory feature database to save.
        db_path: Filesystem path for the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.executescript(_SCHEMA_SQL)

        # -- keyframes & features ------------------------------------------
        for idx, kf in enumerate(db.keyframes):
            pose_blob = kf.camera_pose.astype(np.float64).tobytes()

            # Global descriptor for this keyframe (row from global_descriptors)
            gd_blob: bytes | None = None
            if db.global_descriptors is not None and idx < len(db.global_descriptors):
                gd_blob = db.global_descriptors[idx].astype(np.float64).tobytes()

            cur.execute(
                "INSERT INTO keyframes (id, pose, global_descriptor) VALUES (?, ?, ?)",
                (kf.image_id, pose_blob, gd_blob),
            )

            for j in range(len(kf.keypoints)):
                x, y = kf.keypoints[j]
                x3d, y3d, z3d = kf.points_3d[j]
                desc_blob = kf.descriptors[j].astype(np.uint8).tobytes()
                cur.execute(
                    "INSERT INTO features "
                    "(keyframe_id, x, y, x3d, y3d, z3d, descriptor) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (kf.image_id, float(x), float(y), float(x3d), float(y3d), float(z3d), desc_blob),
                )

        # -- vocabulary ----------------------------------------------------
        if db.vocabulary is not None:
            centers = np.asarray(db.vocabulary.cluster_centers_, dtype=np.float64)
            idf_weights = _compute_idf_weights(db.keyframes, len(centers), centers)

            for word_id in range(len(centers)):
                desc_blob = centers[word_id].tobytes()
                cur.execute(
                    "INSERT INTO vocabulary (word_id, descriptor, idf_weight) "
                    "VALUES (?, ?, ?)",
                    (word_id, desc_blob, float(idf_weights[word_id])),
                )

        conn.commit()
    finally:
        conn.close()


def load_feature_database(db_path: str) -> FeatureDatabase:
    """Load a :class:`FeatureDatabase` from a SQLite file.

    Reads the ``keyframes``, ``features``, and ``vocabulary`` tables and
    reconstructs the in-memory data structures including a scikit-learn
    :class:`~sklearn.cluster.KMeans` vocabulary object.

    Args:
        db_path: Filesystem path to the SQLite database.

    Returns:
        Reconstructed :class:`FeatureDatabase`.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        # -- Load keyframes ------------------------------------------------
        cur.execute("SELECT id, pose, global_descriptor FROM keyframes ORDER BY id")
        kf_rows = cur.fetchall()

        keyframes: List[KeyframeData] = []
        gd_list: list[np.ndarray | None] = []

        for kf_id, pose_blob, gd_blob in kf_rows:
            pose = np.frombuffer(pose_blob, dtype=np.float64).reshape(4, 4).copy()

            gd: np.ndarray | None = None
            if gd_blob is not None:
                gd = np.frombuffer(gd_blob, dtype=np.float64).copy()
            gd_list.append(gd)

            # Load features for this keyframe
            cur.execute(
                "SELECT x, y, x3d, y3d, z3d, descriptor FROM features "
                "WHERE keyframe_id = ? ORDER BY id",
                (kf_id,),
            )
            feat_rows = cur.fetchall()

            kps: List[tuple[float, float]] = []
            pts3d: List[tuple[float, float, float]] = []
            descs: list[np.ndarray] = []

            for x, y, x3d, y3d, z3d, desc_blob in feat_rows:
                kps.append((float(x), float(y)))
                pts3d.append((float(x3d), float(y3d), float(z3d)))
                descs.append(np.frombuffer(desc_blob, dtype=np.uint8).copy())

            desc_array = np.array(descs, dtype=np.uint8) if descs else np.empty((0, 32), dtype=np.uint8)

            keyframes.append(
                KeyframeData(
                    image_id=kf_id,
                    keypoints=kps,
                    descriptors=desc_array,
                    points_3d=pts3d,
                    camera_pose=pose,
                )
            )

        # Build global_descriptors matrix
        global_descriptors: np.ndarray | None = None
        if gd_list and gd_list[0] is not None:
            global_descriptors = np.array(gd_list, dtype=np.float64)

        # -- Load vocabulary -----------------------------------------------
        cur.execute("SELECT word_id, descriptor, idf_weight FROM vocabulary ORDER BY word_id")
        vocab_rows = cur.fetchall()

        vocabulary = None
        if vocab_rows:
            centers = []
            for word_id, desc_blob, idf_weight in vocab_rows:
                centers.append(np.frombuffer(desc_blob, dtype=np.float64).copy())
            centers_array = np.array(centers, dtype=np.float64)

            # Reconstruct a KMeans object with the stored cluster centres
            from sklearn.cluster import KMeans

            k = len(centers_array)
            kmeans = KMeans(n_clusters=k, n_init=1, max_iter=1, random_state=42)
            # Fit on the centres themselves so cluster_centers_ is set correctly
            kmeans.fit(centers_array)
            kmeans.cluster_centers_ = centers_array
            vocabulary = kmeans

        return FeatureDatabase(
            keyframes=keyframes,
            global_descriptors=global_descriptors,
            vocabulary=vocabulary,
        )
    finally:
        conn.close()
