import collections
import itertools

import tensorflow as tf
from deephyper.contrib.layers.stack_mpnn import (
    SPARSE_MPNN,
    GlobalAttentionPool,
    GlobalAttentionSumPool,
    GlobalAvgPool,
    GlobalMaxPool,
    GlobalSumPool,
)
from deephyper.nas.space import KSearchSpace, SpaceFactory
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op import operation
from deephyper.nas.space.op.basic import Zero
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting

SPARSE_MPNN = operation(SPARSE_MPNN)
GlobalAttentionPool = operation(GlobalAttentionPool)
GlobalAttentionSumPool = operation(GlobalAttentionSumPool)
GlobalAvgPool = operation(GlobalAvgPool)
GlobalMaxPool = operation(GlobalMaxPool)
GlobalSumPool = operation(GlobalSumPool)
Flatten = operation(tf.keras.layers.Flatten)
Dense = operation(tf.keras.layers.Dense)

# shapes corresponding to datasets
DATA_SHAPES = {
    "qm7": {
        "input": [
            (8 + 1, 75),
            (8 + 1 + 10 + 1, 2),
            (8 + 1 + 10 + 1, 14),
            (8 + 1,),
            (8 + 1 + 10 + 1,),
        ],
        "output": (1,),
    },
    "qm8": {
        "input": [
            (9 + 1, 75),
            (9 + 1 + 14 + 1, 2),
            (9 + 1 + 14 + 1, 14),
            (9 + 1,),
            (9 + 1 + 14 + 1,),
        ],
        "output": (16,),
    },
    "qm9": {
        "input": [
            (9 + 1, 75),
            (9 + 1 + 16 + 1, 2),
            (9 + 1 + 16 + 1, 14),
            (9 + 1,),
            (9 + 1 + 16 + 1,),
        ],
        "output": (12,),
    },
    "freesolv": {
        "input": [
            (24 + 1, 75),
            (24 + 1 + 25 + 1, 2),
            (24 + 1 + 25 + 1, 14),
            (24 + 1,),
            (24 + 1 + 25 + 1,),
        ],
        "output": (1,),
    },
    "esol": {
        "input": [
            (55 + 1, 75),
            (55 + 1 + 68 + 1, 2),
            (55 + 1 + 68 + 1, 14),
            (55 + 1,),
            (55 + 1 + 68 + 1,),
        ],
        "output": (1,),
    },
    "lipo": {
        "input": [
            (115 + 1, 75),
            (115 + 1 + 236 + 1, 2),
            (115 + 1 + 236 + 1, 14),
            (115 + 1,),
            (115 + 1 + 236 + 1,),
        ],
        "output": (1,),
    },
}


class ResNetMPNNFactory(SpaceFactory):
    def build(
        self,
        input_shape=None,
        output_shape=None,
        num_mpnn_cells=3,
        num_dense_layers=2,
        data="esol",
        **kwargs,
    ):
        """Create a seself.ss space containing multiple Keras self.ssitectures

        Args:
            input_shape (list): the input shapes, e.g. [(3, 4), (5, 2)].
            output_shape (tuple): the output shape, e.g. (12, ).
            num_mpnn_cells (int): the number of MPNN cells.
            num_dense_layers (int): the number of Dense layers.

        Returns:
            A seself.ss space containing multiple Keras self.ssitectures
        """
        input_shape, output_shape = (
            DATA_SHAPES[data]["input"],
            DATA_SHAPES[data]["output"],
        )

        self.ss = KSearchSpace(input_shape, output_shape)
        source = prev_input = self.ss.input_nodes[0]
        prev_input1 = self.ss.input_nodes[1]
        prev_input2 = self.ss.input_nodes[2]
        prev_input3 = self.ss.input_nodes[3]
        prev_input4 = self.ss.input_nodes[4]

        # look over skip connections within a range of the 3 previous nodes
        anchor_points = collections.deque([source], maxlen=3)

        count_gcn_layers = 0
        count_dense_layers = 0
        for _ in range(num_mpnn_cells):
            graph_attn_cell = VariableNode()
            self.mpnn_cell(graph_attn_cell)  #
            self.ss.connect(prev_input, graph_attn_cell)
            self.ss.connect(prev_input1, graph_attn_cell)
            self.ss.connect(prev_input2, graph_attn_cell)
            self.ss.connect(prev_input3, graph_attn_cell)
            self.ss.connect(prev_input4, graph_attn_cell)

            cell_output = graph_attn_cell
            cmerge = ConstantNode()
            cmerge.set_op(AddByProjecting(self.ss, [cell_output], activation="relu"))

            for anchor in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(self.ss, anchor))
                self.ss.connect(skipco, cmerge)

            prev_input = cmerge
            anchor_points.append(prev_input)
            count_gcn_layers += 1

        global_pooling_node = VariableNode()
        self.gather_cell(global_pooling_node)
        self.ss.connect(prev_input, global_pooling_node)
        prev_input = global_pooling_node

        flatten_node = ConstantNode()
        flatten_node.set_op(Flatten())
        self.ss.connect(prev_input, flatten_node)
        prev_input = flatten_node

        for _ in range(num_dense_layers):
            dense_node = ConstantNode()
            dense_node.set_op(Dense(32, activation="relu"))
            self.ss.connect(prev_input, dense_node)
            prev_input = dense_node
            count_dense_layers += 1

        output_node = ConstantNode()
        output_node.set_op(Dense(output_shape[0], activation="linear"))
        self.ss.connect(prev_input, output_node)

        return self.ss

    def mpnn_cell(self, node):
        """Create a variable node of MPNN cell.

        Args:
            node: A DeepHyper variable node object.

        Returns:
            A variable node of MPNN cell.
        """
        state_dims = [4, 8, 16, 32]
        Ts = [1, 2, 3, 4]
        attn_methods = ["const", "gcn", "gat", "sym-gat", "linear", "gen-linear", "cos"]
        attn_heads = [1, 2, 4, 6]
        aggr_methods = ["max", "mean", "sum"]
        update_methods = ["gru", "mlp"]
        activations = [
            tf.keras.activations.sigmoid,
            tf.keras.activations.tanh,
            tf.keras.activations.relu,
            tf.keras.activations.linear,
            tf.keras.activations.elu,
            tf.keras.activations.softplus,
            tf.nn.leaky_relu,
            tf.nn.relu6,
        ]

        for hp in itertools.product(
            state_dims,
            Ts,
            attn_methods,
            attn_heads,
            aggr_methods,
            update_methods,
            activations,
        ):
            (
                state_dim,
                T,
                attn_method,
                attn_head,
                aggr_method,
                update_method,
                activation,
            ) = hp
            node.add_op(
                SPARSE_MPNN(
                    state_dim=state_dim,
                    T=T,
                    attn_method=attn_method,
                    attn_head=attn_head,
                    aggr_method=aggr_method,
                    update_method=update_method,
                    activation=activation,
                )
            )

    def gather_cell(self, node):
        """Create a variable node of Gather cell.

        Args:
            node: A DeepHyper variable node object.

        Returns:
            A variable node of Gather cell.
        """
        for functions in [GlobalSumPool, GlobalMaxPool, GlobalAvgPool]:
            for axis in [-1, -2]:  # Pool in terms of nodes or features
                node.add_op(functions(axis=axis))
        node.add_op(Flatten())
        for state_dim in [16, 32, 64]:
            node.add_op(GlobalAttentionPool(state_dim=state_dim))
        node.add_op(GlobalAttentionSumPool())


def create_search_space(
    input_shape=None,
    output_shape=None,
    num_mpnn_cells=3,
    num_dense_layers=2,
    **kwargs,
):
    return ResNetMPNNFactory()(
        input_shape,
        output_shape,
        num_mpnn_cells=num_mpnn_cells,
        num_dense_layers=num_dense_layers,
        **kwargs,
    )


if __name__ == "__main__":
    data = "esol"
    input_shape, output_shape = DATA_SHAPES[data]["input"], DATA_SHAPES[data]["output"]
    shapes = dict(input_shape=input_shape, output_shape=output_shape)
    factory = ResNetMPNNFactory()
    factory.plot_model(**shapes)