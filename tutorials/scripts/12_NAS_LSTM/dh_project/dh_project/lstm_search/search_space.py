import collections

import tensorflow as tf

from deephyper.nas.space import KSearchSpace, SpaceFactory
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.basic import Zero
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting
from deephyper.nas.space.op.op1d import Identity
from deephyper.nas.space.op import operation

# Convert a Keras layer to a DeepHyper operation
Dense = operation(tf.keras.layers.Dense)
LSTM = operation(tf.keras.layers.LSTM)

class StackedLSTMFactory(SpaceFactory):
    def build(
        self,
        input_shape,
        output_shape,
        num_layers=5,
        **kwargs,
    ):

        self.ss = KSearchSpace(input_shape, output_shape)
        output_dim = output_shape[1]
        source = prev_input = self.ss.input_nodes[0]

        # look over skip connections within a range of the 2 previous nodes
        anchor_points = collections.deque([source], maxlen=2)

        for _ in range(num_layers):
            lstm = VariableNode()
            self.add_lstm_seq_(lstm)
            self.ss.connect(prev_input, lstm)

            cmerge = ConstantNode()
            cmerge.set_op(AddByProjecting(self.ss, [lstm], activation="relu"))

            for anchor in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(self.ss, anchor))
                self.ss.connect(skipco, cmerge)

            # ! for next iter
            prev_input = cmerge
            anchor_points.append(prev_input)

        y = ConstantNode(LSTM(output_dim, return_sequences=True))
        self.ss.connect(prev_input, y)

        return self.ss

    def add_lstm_seq_(self, node):
        node.add_op(Identity())  # we do not want to create a layer in this case
        for units in range(16, 97, 16):
            node.add_op(LSTM(units=units, return_sequences=True))


def create_search_space(
    input_shape=(
        8,
        5,
    ),
    output_shape=(
        8,
        5,
    ),
    num_layers=10,
    **kwargs,
):
    return StackedLSTMFactory()(
        input_shape, output_shape, num_layers=num_layers, **kwargs
    )


if __name__ == "__main__":
    shapes = dict(input_shape=(8, 5,), output_shape=(8, 5,))
    factory = StackedLSTMFactory()
    factory.plot_model(**shapes)