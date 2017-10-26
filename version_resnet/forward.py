from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os

from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf

root = "/scratch/lyc/ai_challenger_scene"

layers = 50

img = load_image(os.path.join(root, "data/cat.jpg"))

sess = tf.Session()

model_dir = os.path.join(root, "model")
new_saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_fn(layers)))
new_saver.restore(sess, os.path.join(model_dir, checkpoint_fn(layers)))

graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")
sub_tensor = graph.get_tensor_by_name("sub:0")

file_path = os.path.join(root, "graph_op_in_forward.txt")
print("writing graph op to file {}".format(file_path))
f = open(file_path, "w+")
for op in graph.get_operations():
    f.write(op.name + "\n")
f.close()

file_path = os.path.join(root, "graph_variables_forward.txt")
print("writing graph variables to file {}".format(file_path))
f = open(file_path, "w+")
list_variables = tf.global_variables()
for v in list_variables:
    f.write(v.name + "\n")
f.close()

file_path = os.path.join(root, "graph_tensor_forward.txt")
print("writing graph tensors to file {}".format(file_path))
f = open(file_path, "w+")
list_variables = tf.get_default_graph().as_graph_def().node
for v in list_variables:
    f.write(v.name + "\n")
f.close()

#init = tf.initialize_all_variables()
#sess.run(init)
print("graph restored")

batch = img.reshape((1, 224, 224, 3))

feed_dict = {images: batch}

prob, sub = sess.run([prob_tensor, sub_tensor], feed_dict=feed_dict)

print("sub_tensor:")
print(sub)

print_prob(prob[0])
