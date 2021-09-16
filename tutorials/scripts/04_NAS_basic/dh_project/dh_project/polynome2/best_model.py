import json
import tensorflow as tf
from dh_project.polynome2.problem import Problem

# Edit the path if necessary
path_to_topk_json = "topk.json"

# Load the json file
with open(path_to_topk_json, "r") as f:
    topk = json.load(f)

# Convert the arch_seq (a str now) to a list
arch_seq = json.loads(topk["0"]["arch_seq"])

# Create the Keras model using the problem
model = Problem.get_keras_model(arch_seq)
tf.keras.utils.plot_model(model, "best_model.png")