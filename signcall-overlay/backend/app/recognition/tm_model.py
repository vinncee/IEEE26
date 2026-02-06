"""Teachable Machine Pose model — PoseNet feature extraction + Dense classifier.

Pipeline:
    1. Resize input BGR frame to 257×257
    2. Run PoseNet MobileNetV1 to get heatmaps (17×17×17) + offsets (17×17×34)
    3. Flatten to 14739-dim feature vector
    4. Feed into the TM Dense classifier (Dense→Dropout→Dense)
    5. Return (label, confidence) for the best-matching class

The PoseNet backbone and TM classifier are loaded once at import time.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from google.protobuf import json_format
from tensorflow.core.framework import graph_pb2

# ── Paths ──────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_TM_DIR = _BASE_DIR / "tm_model"
_POSENET_DIR = _TM_DIR / "posenet"
_POSENET_SAVED = _TM_DIR / "posenet_saved"
_CLASSIFIER_H5 = _TM_DIR / "tm_classifier.h5"
_METADATA_JSON = _TM_DIR / "metadata.json"

# ── PoseNet constants ─────────────────────────────────────────────────────
_INPUT_SIZE = 257  # PoseNet MobileNetV1 input resolution
_FEATURE_DIM = 14739  # 17×17×17 heatmaps + 17×17×34 offsets

# ── Label mapping  (TM label → internal token) ───────────────────────────
# Teachable Machine labels may use natural language; our system uses
# uppercase tokens.  This map is built from metadata.json at load time.
_TM_LABEL_TO_TOKEN = {
    "Thank You": "THANKS",
    "How": "HOW",
    "You": "YOU",
    "Can": "CAN",
    "Slow": "SLOW",
    "Repeat": "REPEAT",
    "Hello": "HELLO",
    "see you later": "SEE_YOU_LATER",
    "father": "FATHER",
    "Mother": "MOTHER",
}

# ── Module-level singletons (lazy-loaded) ─────────────────────────────────
_posenet_session: Optional[tf.compat.v1.Session] = None
_posenet_input = None
_posenet_heatmap = None
_posenet_offset = None
_classifier: Optional[tf.keras.Model] = None
_labels: list[str] = []
_tokens: list[str] = []


# ═══════════════════════════════════════════════════════════════════════════
# Loader helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_posenet_from_tfjs() -> tuple:
    """Load PoseNet from raw TF.js graph model files (model.json + shards)."""
    model_json_path = _POSENET_DIR / "model.json"
    with open(model_json_path) as f:
        model_json = json.load(f)

    topology = model_json["modelTopology"]
    manifest = model_json["weightsManifest"][0]

    # Parse GraphDef from JSON
    graph_def = json_format.ParseDict(topology, graph_pb2.GraphDef())

    # Load weight shards
    weight_data = bytearray()
    for shard_path in manifest["paths"]:
        with open(_POSENET_DIR / shard_path, "rb") as f:
            weight_data.extend(f.read())

    # Parse individual weight arrays
    weights = {}
    offset = 0
    for spec in manifest["weights"]:
        shape = spec["shape"]
        n = int(np.prod(shape))
        nbytes = n * 4
        arr = np.frombuffer(
            bytes(weight_data[offset : offset + nbytes]), dtype=np.float32
        ).reshape(shape)
        weights[spec["name"]] = arr
        offset += nbytes

    # Inject weights into Const nodes
    for node in graph_def.node:
        if node.op == "Const" and node.name in weights:
            tensor = tf.make_tensor_proto(weights[node.name])
            node.attr["value"].tensor.CopyFrom(tensor)

    # Build graph
    g = tf.compat.v1.Graph()
    with g.as_default():
        tf.graph_util.import_graph_def(graph_def, name="")

    sess = tf.compat.v1.Session(graph=g)
    inp = g.get_tensor_by_name("sub_2:0")
    hm = g.get_tensor_by_name("MobilenetV1/heatmap_2/BiasAdd:0")
    off = g.get_tensor_by_name("MobilenetV1/offset_2/BiasAdd:0")
    return sess, inp, hm, off


def _load_posenet_from_saved() -> tuple:
    """Load PoseNet from a previously-exported SavedModel (faster)."""
    g = tf.compat.v1.Graph()
    sess = tf.compat.v1.Session(graph=g)
    tf.compat.v1.saved_model.loader.load(
        sess, [tf.saved_model.SERVING], str(_POSENET_SAVED)
    )
    inp = g.get_tensor_by_name("sub_2:0")
    hm = g.get_tensor_by_name("MobilenetV1/heatmap_2/BiasAdd:0")
    off = g.get_tensor_by_name("MobilenetV1/offset_2/BiasAdd:0")
    return sess, inp, hm, off


def _ensure_loaded():
    """Lazy-load PoseNet backbone + TM classifier on first call."""
    global _posenet_session, _posenet_input, _posenet_heatmap, _posenet_offset
    global _classifier, _labels, _tokens

    if _posenet_session is not None:
        return  # already loaded

    logger.info("Loading Teachable Machine pose model...")

    # 1. Load PoseNet backbone (prefer SavedModel if available)
    if _POSENET_SAVED.exists():
        logger.info("Loading PoseNet from SavedModel...")
        _posenet_session, _posenet_input, _posenet_heatmap, _posenet_offset = (
            _load_posenet_from_saved()
        )
    else:
        logger.info("Loading PoseNet from TF.js files...")
        _posenet_session, _posenet_input, _posenet_heatmap, _posenet_offset = (
            _load_posenet_from_tfjs()
        )

    # 2. Load TM classifier head
    _classifier = tf.keras.models.load_model(str(_CLASSIFIER_H5), compile=False)
    logger.info("TM classifier loaded: %s", _classifier.input_shape)

    # 3. Load labels from metadata.json
    with open(_METADATA_JSON) as f:
        meta = json.load(f)
    _labels = meta["labels"]
    _tokens = [_TM_LABEL_TO_TOKEN.get(lbl, lbl.upper()) for lbl in _labels]
    logger.info("TM labels: %s", _labels)
    logger.info("TM tokens: %s", _tokens)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def is_available() -> bool:
    """Check whether all model files exist (without loading them)."""
    return (
        _CLASSIFIER_H5.exists()
        and _METADATA_JSON.exists()
        and (_POSENET_SAVED.exists() or (_POSENET_DIR / "model.json").exists())
    )


def _pad_and_resize(frame_rgb: np.ndarray) -> np.ndarray:
    """Replicate Teachable Machine / PoseNet's ``padAndResizeTo`` exactly.

    TM pads the input image to make it square (preserving aspect ratio)
    THEN resizes to 257×257.  A raw ``cv2.resize`` would stretch a 16:9
    webcam frame into a square, distorting the pose and producing wrong
    heatmaps.

    Reference: tensorflow/tfjs-models posenet/src/util.ts  padAndResizeTo()
    """
    h, w = frame_rgb.shape[:2]
    target_h, target_w = _INPUT_SIZE, _INPUT_SIZE
    target_aspect = target_w / target_h  # 1.0 for 257×257
    aspect = w / h

    if aspect < target_aspect:
        # Image is taller than wide → pad width (left + right)
        pad_l = round(0.5 * (target_aspect * h - w))
        pad_r = pad_l
        pad_t = 0
        pad_b = 0
    else:
        # Image is wider than tall → pad height (top + bottom)
        pad_t = round(0.5 * ((1.0 / target_aspect) * w - h))
        pad_b = pad_t
        pad_l = 0
        pad_r = 0

    # Pad with zeros (black) — matches tf.pad3d default
    padded = cv2.copyMakeBorder(
        frame_rgb, pad_t, pad_b, pad_l, pad_r,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )

    # Resize (bilinear interpolation — matches tf.image.resizeBilinear)
    resized = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized


def extract_posenet_features(frame_bgr: np.ndarray) -> np.ndarray:
    """Run PoseNet on a BGR frame and return the 14739-dim feature vector.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR image (any size) — will be padded + resized to 257×257.

    Returns
    -------
    np.ndarray of shape (14739,)
    """
    _ensure_loaded()

    # Preprocess: BGR→RGB, pad-and-resize (preserving aspect ratio), normalise
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = _pad_and_resize(rgb)
    img = img.astype(np.float32) / 127.5 - 1.0  # normalise to [-1, 1]
    img = np.expand_dims(img, axis=0)  # (1, 257, 257, 3)

    # Run PoseNet backbone
    heatmaps, offsets = _posenet_session.run(
        [_posenet_heatmap, _posenet_offset],
        feed_dict={_posenet_input: img},
    )

    # CRITICAL: PoseNet's baseModel.predict() applies sigmoid to the raw
    # heatmap logits before returning them as `heatmapScores`.  Teachable
    # Machine receives sigmoid-activated values — not raw logits — so the
    # TM classifier was trained on sigmoid(heatmap) features.
    #   JS:  heatmapScores = tf.sigmoid(namedResults.heatmap)
    # See: tensorflow/tfjs-models  posenet/src/base_model.ts  predict()
    heatmaps = 1.0 / (1.0 + np.exp(-heatmaps))   # sigmoid

    # Teachable Machine concatenates 3D tensors along the CHANNEL axis
    # (axis=2) BEFORE flattening.  This produces a different element
    # ordering than flattening each tensor independently then concatenating.
    #
    # TM JS: tf.concat([heatmapScores, offsets], axis=2)  →  [17,17,51]  →  flatten
    # We must replicate the same order.
    hm3d = heatmaps[0]   # (17, 17, 17)  — now sigmoid-activated
    off3d = offsets[0]    # (17, 17, 34)
    concat_3d = np.concatenate([hm3d, off3d], axis=2)  # (17, 17, 51)
    features = concat_3d.flatten()  # 14739
    assert features.shape[0] == _FEATURE_DIM, (
        f"Expected {_FEATURE_DIM} features, got {features.shape[0]}"
    )
    return features


def predict_frame(frame_bgr: np.ndarray) -> dict:
    """Classify a single BGR frame using the TM pose model.

    Returns
    -------
    dict with keys: token, confidence, top2, probabilities
    """
    _ensure_loaded()

    features = extract_posenet_features(frame_bgr)
    features_batch = features.reshape(1, -1)

    probs = _classifier.predict(features_batch, verbose=0)[0]

    sorted_indices = np.argsort(probs)[::-1]
    top_idx = sorted_indices[0]
    second_idx = sorted_indices[1]

    # Log top-3 probabilities for debugging
    top3 = [(_tokens[sorted_indices[i]], float(probs[sorted_indices[i]])) for i in range(min(3, len(sorted_indices)))]
    logger.info("TM probs top3: %s", "  ".join(f"{t}={p:.3f}" for t, p in top3))

    return {
        "token": _tokens[top_idx],
        "confidence": float(probs[top_idx]),
        "top2": [_tokens[top_idx], _tokens[second_idx]],
        "probabilities": {_tokens[i]: float(probs[i]) for i in range(len(probs))},
    }


def predict_window(frames_bgr: list[np.ndarray]) -> dict:
    """Classify using the most recent BGR frame (matching TM browser behavior).

    Teachable Machine classifies one frame at a time — it does NOT aggregate
    over a window.  Using the last (most recent) frame ensures the prediction
    reflects the user's current pose without lag.

    Parameters
    ----------
    frames_bgr : list of BGR numpy arrays

    Returns
    -------
    dict with keys: token, confidence, top2
    """
    _ensure_loaded()

    if not frames_bgr:
        return {"token": "HELLO", "confidence": 0.0, "top2": ["HELLO", "HELLO"]}

    # Use the LAST (most recent) frame — matches TM's real-time behavior
    return predict_frame(frames_bgr[-1])
