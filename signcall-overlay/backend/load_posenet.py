"""Load PoseNet TF.js graph model directly in Python.

Reads model.json (GraphDef as JSON) + binary weight shards,
builds a TF graph, and runs inference to produce the 14739-dim
feature vector that the Teachable Machine classifier expects.
"""

import os
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from google.protobuf import json_format
from tensorflow.core.framework import graph_pb2

POSENET_DIR = os.path.join(os.path.dirname(__file__), "tm_model", "posenet")


def main():
    # 1. Load model.json
    with open(os.path.join(POSENET_DIR, "model.json")) as f:
        model_json = json.load(f)

    topology = model_json["modelTopology"]
    manifest = model_json["weightsManifest"][0]

    # 2. Convert JSON topology to GraphDef protobuf
    graph_def = json_format.ParseDict(topology, graph_pb2.GraphDef())
    print(f"GraphDef has {len(graph_def.node)} nodes")

    # 3. Load weight shards into a single byte buffer
    weight_data = bytearray()
    for shard_path in manifest["paths"]:
        full_path = os.path.join(POSENET_DIR, shard_path)
        with open(full_path, "rb") as f:
            weight_data.extend(f.read())
    print(f"Total weight bytes: {len(weight_data):,}")

    # 4. Parse weights according to manifest
    weight_specs = manifest["weights"]
    offset = 0
    weights = {}
    for spec in weight_specs:
        name = spec["name"]
        shape = spec["shape"]
        dtype = np.float32
        n = int(np.prod(shape))
        nbytes = n * 4
        arr = np.frombuffer(
            bytes(weight_data[offset : offset + nbytes]), dtype=dtype
        ).reshape(shape)
        weights[name] = arr
        offset += nbytes
    print(f"Loaded {len(weights)} weight tensors")

    # 5. Inject actual weight values into Const nodes of the GraphDef
    for node in graph_def.node:
        if node.op == "Const" and node.name in weights:
            tensor = tf.make_tensor_proto(weights[node.name])
            node.attr["value"].tensor.CopyFrom(tensor)

    # 6. Import graph
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")

    # 7. Find input/output tensors
    input_tensor = graph.get_tensor_by_name("sub_2:0")
    heatmap_tensor = graph.get_tensor_by_name("MobilenetV1/heatmap_2/BiasAdd:0")
    offset_tensor = graph.get_tensor_by_name("MobilenetV1/offset_2/BiasAdd:0")

    print(f"\nInput tensor: {input_tensor.name}  shape={input_tensor.shape}")
    print(f"Heatmap tensor: {heatmap_tensor.name}  shape={heatmap_tensor.shape}")
    print(f"Offset tensor: {offset_tensor.name}  shape={offset_tensor.shape}")

    # 8. Test with dummy 257x257 RGB image
    with tf.compat.v1.Session(graph=graph) as sess:
        dummy = np.random.rand(1, 257, 257, 3).astype(np.float32)
        heatmaps, offsets = sess.run(
            [heatmap_tensor, offset_tensor],
            feed_dict={input_tensor: dummy},
        )
        print(f"\nHeatmaps shape: {heatmaps.shape}")
        print(f"Offsets shape:  {offsets.shape}")
        features = np.concatenate([heatmaps.flatten(), offsets.flatten()])
        print(f"Feature vector dim: {features.shape[0]}")
        print(f"Expected: 14739, Match: {features.shape[0] == 14739}")

    # 9. Save as SavedModel for faster loading later
    saved_model_dir = os.path.join(os.path.dirname(__file__), "tm_model", "posenet_saved")
    if not os.path.exists(saved_model_dir):
        with tf.compat.v1.Session(graph=graph) as sess:
            tf.compat.v1.saved_model.simple_save(
                sess,
                saved_model_dir,
                inputs={"image": input_tensor},
                outputs={
                    "heatmaps": heatmap_tensor,
                    "offsets": offset_tensor,
                },
            )
        print(f"\n✅ Saved PoseNet as SavedModel to {saved_model_dir}")
    else:
        print(f"\nSavedModel already exists at {saved_model_dir}")

    print("\n✅ PoseNet loaded and working!")


if __name__ == "__main__":
    main()
