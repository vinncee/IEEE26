"""Verify that PoseNet heatmap needs sigmoid activation."""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

# Load PoseNet
g = tf.compat.v1.Graph()
sess = tf.compat.v1.Session(graph=g)
tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.SERVING], 'tm_model/posenet_saved')

inp = g.get_tensor_by_name('sub_2:0')
hm = g.get_tensor_by_name('MobilenetV1/heatmap_2/BiasAdd:0')
off = g.get_tensor_by_name('MobilenetV1/offset_2/BiasAdd:0')

# Run on a test image
test_img = np.random.rand(1, 257, 257, 3).astype(np.float32) * 2 - 1
heatmaps, offsets = sess.run([hm, off], feed_dict={inp: test_img})
print('Heatmap shape:', heatmaps.shape, 'min:', heatmaps.min(), 'max:', heatmaps.max())
print('Offsets shape:', offsets.shape, 'min:', offsets.min(), 'max:', offsets.max())

# Apply sigmoid to heatmaps (as PoseNet does in JS)
hm_sigmoid = 1.0 / (1.0 + np.exp(-heatmaps))
print()
print('Heatmap after sigmoid min:', hm_sigmoid.min(), 'max:', hm_sigmoid.max())
print()
print('Sample raw heatmap values:', heatmaps[0, 8, 8, :5])
print('Sample sigmoid values:    ', hm_sigmoid[0, 8, 8, :5])

# Show the impact: without sigmoid the features are drastically different
hm3d = heatmaps[0]       # (17, 17, 17)
off3d = offsets[0]        # (17, 17, 34)
concat_raw = np.concatenate([hm3d, off3d], axis=2).flatten()

hm3d_sig = hm_sigmoid[0]  # (17, 17, 17)
concat_sig = np.concatenate([hm3d_sig, off3d], axis=2).flatten()

print()
print('Feature vector (RAW heatmap) first 10:', concat_raw[:10])
print('Feature vector (SIGMOID heatmap) first 10:', concat_sig[:10])
print()
print('Feature L2 norm (raw):', np.linalg.norm(concat_raw))
print('Feature L2 norm (sigmoid):', np.linalg.norm(concat_sig))
