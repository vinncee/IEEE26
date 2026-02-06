"""Verify the TF.js → Keras H5 weight conversion and test inference directly."""
import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# 1. Load original TF.js weights directly from weights.bin
with open('tm_model/model.json') as f:
    mj = json.load(f)
manifest = mj['weightsManifest'][0]['weights']

with open('tm_model/weights.bin', 'rb') as f:
    raw = f.read()

offset = 0
orig_weights = {}
for spec in manifest:
    shape = spec['shape']
    n = int(np.prod(shape))
    nbytes = n * 4
    arr = np.frombuffer(raw[offset:offset+nbytes], dtype=np.float32).reshape(shape)
    orig_weights[spec['name']] = arr.copy()
    offset += nbytes
    print(f'TF.js weight: {spec["name"]}  shape={arr.shape}')

# 2. Load converted H5 model
h5 = tf.keras.models.load_model('tm_model/tm_classifier.h5', compile=False)
print()
h5.summary()

# 3. Compare weights
print("\n=== Weight Comparison ===")
for layer in h5.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        keras_w = layer.get_weights()
        tfjs_key = f'{layer.name}/kernel'
        tfjs_kernel = orig_weights[tfjs_key]
        print(f'\nLayer {layer.name}:')
        print(f'  Keras kernel shape: {keras_w[0].shape}  TF.js kernel shape: {tfjs_kernel.shape}')
        diff = np.abs(keras_w[0] - tfjs_kernel).max()
        print(f'  Kernel max diff: {diff}')
        print(f'  Kernels identical? {np.allclose(keras_w[0], tfjs_kernel)}')
        
        bias_key = f'{layer.name}/bias'
        if len(keras_w) > 1:
            tfjs_bias = orig_weights[bias_key]
            diff_b = np.abs(keras_w[1] - tfjs_bias).max()
            print(f'  Bias max diff: {diff_b}')
        else:
            print(f'  No bias in Keras')
            if bias_key in orig_weights:
                print(f'  WARNING: TF.js HAS bias but Keras does not!')

# 4. Test: Run inference using raw numpy matmul vs Keras model.predict
print("\n=== Inference Comparison (random input) ===")
np.random.seed(42)
test_input = np.random.randn(1, 14739).astype(np.float32)

# Keras inference
keras_out = h5.predict(test_input, verbose=0)
print(f'Keras output: {keras_out[0][:5]}...')

# Manual numpy inference (matching TF.js behavior)
k1 = orig_weights['dense_Dense1/kernel']
b1 = orig_weights['dense_Dense1/bias']
k2 = orig_weights['dense_Dense2/kernel']

# Dense1: matmul + bias + relu
z1 = test_input @ k1 + b1
a1 = np.maximum(z1, 0)  # ReLU
# No dropout at inference
# Dense2: matmul (no bias) + softmax
z2 = a1 @ k2
exp_z2 = np.exp(z2 - z2.max())
numpy_out = exp_z2 / exp_z2.sum()
print(f'NumPy output: {numpy_out[0][:5]}...')
print(f'Outputs match? {np.allclose(keras_out, numpy_out, atol=1e-5)}')
print(f'Max diff: {np.abs(keras_out - numpy_out).max()}')

# 5. Load labels
with open('tm_model/metadata.json') as f:
    meta = json.load(f)
labels = meta['labels']

print(f'\nKeras top prediction: {labels[np.argmax(keras_out)]} ({keras_out[0][np.argmax(keras_out)]:.4f})')
print(f'NumPy top prediction: {labels[np.argmax(numpy_out)]} ({numpy_out[0][np.argmax(numpy_out)]:.4f})')

# 6. Check if Dense2 in model.json has use_bias=false
layers = mj['modelTopology']['config']['layers']
for l in layers:
    if l['class_name'] == 'Dense':
        print(f"\nLayer {l['config']['name']}: use_bias={l['config'].get('use_bias', True)}")
