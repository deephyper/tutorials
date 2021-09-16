import collections

import tensorflow as tf

from deephyper.nas.space import KSearchSpace, SpaceFactory
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op import operation
from deephyper.nas.space.op.basic import Zero
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting
from deephyper.nas.space.op.op1d import Identity

Dense = operation(tf.keras.layers.Dense)
Concatenate = operation(tf.keras.layers.Concatenate)


class MultiInputsResNetMLPFactory(SpaceFactory):
    def build(
        self, input_shape=[(2,), (1,), (1,)], output_shape=(1,), num_layers=10, **kwargs
    ):

        assert len(input_shape) == 3

        self.ss = KSearchSpace(input_shape, output_shape)

        # Three input tensors
        input_0, input_1, input_2 = self.ss.input_nodes

        concat = ConstantNode(Concatenate())
        self.ss.connect(input_0, concat)
        self.ss.connect(input_1, concat)
        self.ss.connect(input_2, concat)

        # Input anchors (recorded so they can be connected to anywhere
        # in the architecture)
        input_anchors = [input_1, input_2]

        # Creates a Queue to store outputs of the 3 previously created  layers
        # to create potential residual connections
        skip_anchors = collections.deque([input_0], maxlen=3)

        prev_input = concat
        for _ in range(num_layers):
            dense = VariableNode()
            self.add_dense_to_(dense)

            self.ss.connect(prev_input, dense)

            # ConstantNode to merge possible residual connections from the different
            # input tensors (input_0, input_1, input_2)
            merge_0 = ConstantNode()
            merge_0.set_op(AddByProjecting(self.ss, [dense], activation="relu"))

            # Creates potential connections to the various input tensors
            for anchor in input_anchors:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(self.ss, anchor))
                self.ss.connect(skipco, merge_0)

            # ConstantNode to merge possible
            merge_1 = ConstantNode()
            merge_1.set_op(AddByProjecting(self.ss, [merge_0], activation="relu"))

            # a potential connection to the variable nodes (vnodes) of the previous layers
            for anchor in skip_anchors:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(self.ss, anchor))
                self.ss.connect(skipco, merge_1)

            # ! for next iter
            prev_input = merge_1
            skip_anchors.append(prev_input)

        output_node = ConstantNode(Dense(output_shape[0]))
        self.ss.connect(prev_input, output_node)

        return self.ss

    def add_dense_to_(self, node):
        node.add_op(Identity())  # we do not want to create a layer in this case

        activations = [
            tf.keras.activations.linear,
            tf.keras.activations.relu,
            tf.keras.activations.tanh,
            tf.keras.activations.sigmoid,
        ]
        for units in range(16, 97, 16):
            for activation in activations:
                node.add_op(Dense(units=units, activation=activation))


def create_search_space(input_shape=[(2,), (1,), (1,)], output_shape=(1,), **kwargs):
    return MultiInputsResNetMLPFactory()(input_shape, output_shape, **kwargs)


if __name__ == "__main__":
    shapes = dict(input_shape=[(2,), (1,), (1,)], output_shape=(1,))
    factory = MultiInputsResNetMLPFactory()
    factory.plot_model(**shapes, num_layers=4)