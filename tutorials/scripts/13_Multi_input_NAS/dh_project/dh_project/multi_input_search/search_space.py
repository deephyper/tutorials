import collections

import tensorflow as tf

from deephyper.nas.space import AutoKSearchSpace
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.basic import Tensor
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting
from deephyper.nas.space.op.op1d import Dense, Identity


def add_dense_to_(node):
    node.add_op(Identity()) # we do not want to create a layer in this case

    activations = [None, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for units in range(16, 97, 16):
        for activation in activations:
            node.add_op(Dense(units=units, activation=activation))


def create_search_space(input_shape=[(2,),(1,),(1,)],
                        output_shape=(1,),
                        num_layers=10,
                        *args, **kwargs):

    arch = AutoKSearchSpace(input_shape, output_shape, regression=True)

    # Three input tensors
    source_0 = arch.input_nodes[0]
    source_1 = arch.input_nodes[1]
    source_2 = arch.input_nodes[2]

    # input anchors (recorded so they can be connected to anywhere in the architecture)
    input_anchors = collections.deque([source_1,source_2], maxlen=2)

    # look over skip connections within a range of the 3 previous nodes
    skip_anchors = collections.deque([source_0], maxlen=3)
    prev_input = source_0

    for _ in range(num_layers):
        vnode = VariableNode()
        add_dense_to_(vnode)

        arch.connect(prev_input, vnode)

        # * Cell output
        cell_output = vnode

        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(arch, [cell_output], activation='relu'))

        # a potential connection to the various input tensors
        for anchor in input_anchors:
            skipco = VariableNode()
            skipco.add_op(Tensor([]))
            skipco.add_op(Connect(arch, anchor))
            arch.connect(skipco, cmerge)

        # ! for next iter
        prev_input = cmerge

        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(arch, [prev_input], activation='relu'))

        # a potential connection to the variable nodes (vnodes) of the previous layers
        for anchor in skip_anchors:
            skipco = VariableNode()
            skipco.add_op(Tensor([]))
            skipco.add_op(Connect(arch, anchor))
            arch.connect(skipco, cmerge)

        # ! for next iter
        prev_input = cmerge
        skip_anchors.append(prev_input)

    return arch


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf

    search_space = create_search_space(num_layers=3)
    ops = [random() for _ in range(search_space.num_nodes)]

    print(f'This search_space needs {len(ops)} choices to generate a neural network.')

    search_space.set_ops(ops)

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file='sampled_neural_network.png', show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")


if __name__ == '__main__':
    test_create_search_space()