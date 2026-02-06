"""Convert Teachable Machine TF.js Layers model to Keras H5 format.

This avoids the tensorflowjs_converter CLI (which has protobuf conflicts)
by using tensorflowjs Python API directly with targeted imports.

Usage:  python convert_tm_model.py
"""

import os
import json
import struct
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings

import tensorflow as tf

MODEL_DIR = os.path.join(os.path.dirname(__file__), "tm_model")
MODEL_JSON = os.path.join(MODEL_DIR, "model.json")
WEIGHTS_BIN = os.path.join(MODEL_DIR, "weights.bin")
METADATA_JSON = os.path.join(MODEL_DIR, "metadata.json")
OUTPUT_H5 = os.path.join(MODEL_DIR, "tm_classifier.h5")


def load_weights_from_bin(model_json: dict, weights_path: str) -> list:
    """Read binary weights from weights.bin according to model.json manifest."""
    # model.json has weightsManifest[0].weights — list of {name, shape, dtype}
    manifest = model_json["weightsManifest"][0]["weights"]

    with open(weights_path, "rb") as f:
        raw = f.read()

    offset = 0
    arrays = []
    for spec in manifest:
        name = spec["name"]
        shape = spec["shape"]
        dtype_str = spec.get("dtype", "float32")

        # Map TF.js dtype to numpy dtype
        np_dtype = {"float32": np.float32, "int32": np.int32}[dtype_str]
        n_elements = 1
        for s in shape:
            n_elements *= s
        n_bytes = n_elements * np_dtype().itemsize

        arr = np.frombuffer(raw[offset : offset + n_bytes], dtype=np_dtype).reshape(shape)
        arrays.append(arr)
        offset += n_bytes
        print(f"  Loaded weight '{name}' shape={shape} dtype={dtype_str}")

    return arrays


def main():
    # Load model.json
    with open(MODEL_JSON) as f:
        model_json = json.load(f)

    with open(METADATA_JSON) as f:
        metadata = json.load(f)

    labels = metadata["labels"]
    print(f"Labels ({len(labels)}): {labels}")

    # Parse topology
    config = model_json["modelTopology"]["config"]
    layers = config["layers"]
    print(f"Model has {len(layers)} layers:")
    for layer in layers:
        cls = layer["class_name"]
        lconf = layer["config"]
        name = lconf.get("name", "?")
        if cls == "Dense":
            print(f"  {cls}({name}): units={lconf['units']}, activation={lconf['activation']}")
        elif cls == "Dropout":
            print(f"  {cls}({name}): rate={lconf['rate']}")
        else:
            print(f"  {cls}({name})")

    # Determine input shape from first Dense layer
    first_dense = [l for l in layers if l["class_name"] == "Dense"][0]
    batch_input_shape = first_dense["config"].get("batch_input_shape", [None, 14739])
    input_dim = batch_input_shape[-1]
    print(f"\nInput dimension: {input_dim}")

    # Load binary weights
    weight_arrays = load_weights_from_bin(model_json, WEIGHTS_BIN)

    # Build a name→array map from loaded weights
    manifest = model_json["weightsManifest"][0]["weights"]
    weight_map = {}
    for i, spec in enumerate(manifest):
        weight_map[spec["name"]] = weight_arrays[i]
    print(f"\nWeight map keys: {list(weight_map.keys())}")

    # Build equivalent Keras model
    model = tf.keras.Sequential()
    for layer in layers:
        cls = layer["class_name"]
        lconf = layer["config"]
        if cls == "Dense":
            units = lconf["units"]
            activation = lconf["activation"]
            use_bias = lconf.get("use_bias", True)
            # Check if bias weight actually exists
            layer_name = lconf["name"]
            has_bias_weight = f"{layer_name}/bias" in weight_map
            actual_use_bias = use_bias and has_bias_weight

            if lconf.get("batch_input_shape"):
                model.add(tf.keras.layers.Dense(
                    units, activation=activation,
                    use_bias=actual_use_bias,
                    input_shape=(input_dim,),
                    name=layer_name
                ))
            else:
                model.add(tf.keras.layers.Dense(
                    units, activation=activation,
                    use_bias=actual_use_bias,
                    name=layer_name
                ))
        elif cls == "Dropout":
            model.add(tf.keras.layers.Dropout(lconf["rate"], name=lconf["name"]))

    # Build model by running dummy input
    dummy = np.zeros((1, input_dim), dtype=np.float32)
    model(dummy)

    # Set weights from the weight map
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_name = layer.name
            kernel = weight_map[f"{layer_name}/kernel"]
            if f"{layer_name}/bias" in weight_map:
                bias = weight_map[f"{layer_name}/bias"]
                layer.set_weights([kernel, bias])
            else:
                layer.set_weights([kernel])

    model.summary()

    # Save as H5
    model.save(OUTPUT_H5)
    print(f"\n✅ Saved Keras model to {OUTPUT_H5}")
    print(f"   Input: ({input_dim},) → Output: ({len(labels)},)")

    # Quick test
    test_input = np.random.randn(1, input_dim).astype(np.float32)
    preds = model.predict(test_input, verbose=0)
    top_idx = np.argmax(preds[0])
    print(f"\n   Test prediction: {labels[top_idx]} ({preds[0][top_idx]:.4f})")
    print(f"   All probs: {dict(zip(labels, [f'{p:.4f}' for p in preds[0]]))}")


if __name__ == "__main__":
    main()
