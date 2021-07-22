import collections

import tensorflow as tf

from deephyper.nas.space import KSearchSpace, AutoKSearchSpace
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.basic import Tensor
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting, AddByPadding, Concatenate
from deephyper.nas.space.op.op1d import Dense, Identity


def add_dense_to_(node):
    node.add_op(Identity()) # we do not want to create a layer in this case

    activations = [None, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for units in range(16, 97, 16):
        for activation in activations:
            node.add_op(Dense(units=units, activation=activation))

def add_lstm_seq_(node):
    node.add_op(Identity()) # we do not want to create a layer in this case
    #activations = [None, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for units in range(16, 97, 16):
        node.add_op(tf.keras.layers.LSTM(units=units, return_sequences=True))

def add_lstm_oplayer_(node,units):
    node.set_op(tf.keras.layers.LSTM(units=units, return_sequences=True))

def add_lstm_(node):
    node.add_op(Identity()) # we do not want to create a layer in this case
    #activations = [None, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for units in range(16, 97, 16):
        node.add_op(tf.keras.layers.LSTM(units=units, return_sequences=False))


def create_search_space(input_shape=(8,5,),
                        output_shape=(8,5,),
                        num_layers=10,
                        *args, **kwargs):

    arch = KSearchSpace(input_shape, output_shape, regression=True)
    source = prev_input = arch.input_nodes[0]

    # look over skip connections within a range of the 2 previous nodes
    anchor_points = collections.deque([source], maxlen=2)

    for _ in range(num_layers):
        vnode = VariableNode()
        add_lstm_seq_(vnode)

        arch.connect(prev_input, vnode)

        # * Cell output
        cell_output = vnode

        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(arch, [cell_output], activation='relu'))
        # cmerge.set_op(Concatenate(arch, [cell_output]))

        for anchor in anchor_points:
            skipco = VariableNode()
            skipco.add_op(Tensor([]))
            skipco.add_op(Connect(arch, anchor))
            arch.connect(skipco, cmerge)

        # ! for next iter
        prev_input = cmerge
        anchor_points.append(prev_input)

        # prev_input = cell_output
    cnode = ConstantNode()
    add_lstm_oplayer_(cnode,5)
    arch.connect(prev_input,cnode)

    return arch


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    import random
    random.seed(10)

    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    tf.random.set_seed(10)

    search_space = create_search_space(num_layers=5)
    ops = [random() for _ in range(search_space.num_nodes)]
    search_space.set_ops(ops)
    model = search_space.create_model()
    model.summary()

    print(f'This search_space needs {len(ops)} choices to generate a neural network.')


if __name__ == '__main__':
    test_create_search_space()