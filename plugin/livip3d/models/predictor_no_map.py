from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .predictor_decoder import Decoder, DecoderResCat
from .predictor_lib import MLP, GlobalGraph, LayerNorm, SubGraph, CrossAttention, GlobalGraphRes
from .. import utils as utils


class NewSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=3):
        super(NewSubGraph, self).__init__()
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])
        if True:
            self.layer_0 = MLP(hidden_size)
            self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
            self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
            self.layers_3 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
            self.layers_4 = nn.ModuleList([GlobalGraph(hidden_size) for _ in range(depth)])
            if True:
                self.layer_0_again = MLP(hidden_size)

    def forward(self, input_list: list):
        batch_size = len(input_list)
        device = input_list[0].device
        hidden_states, lengths = utils.merge_tensors(input_list, device)
        hidden_size = hidden_states.shape[2]
        max_vector_num = hidden_states.shape[1]

        if True:
            attention_mask = torch.zeros([batch_size, max_vector_num, max_vector_num], device=device)
            hidden_states = self.layer_0(hidden_states)

            if True:
                hidden_states = self.layer_0_again(hidden_states)
            for i in range(batch_size):
                assert lengths[i] > 0
                attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

            for layer_index, layer in enumerate(self.layers):
                temp = hidden_states
                # hidden_states = layer(hidden_states, attention_mask)
                # hidden_states = self.layers_2[layer_index](hidden_states)
                # hidden_states = F.relu(hidden_states) + temp
                hidden_states = layer(hidden_states, attention_mask)
                hidden_states = F.relu(hidden_states)
                hidden_states = hidden_states + temp
                hidden_states = self.layers_2[layer_index](hidden_states)

        return torch.max(hidden_states, dim=1)[0], torch.cat(utils.de_merge_tensors(hidden_states, lengths))


class CustomGraph(nn.Module):
    r"""
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """

    def __init__(self,
                 hidden_size=128,
                 decoder=None,
                 ):
        super(CustomGraph, self).__init__()

        self.sub_graph = SubGraph(hidden_size)

        if True:
            self.point_level_sub_graph = NewSubGraph(hidden_size)
            # self.point_level_cross_attention = CrossAttention(hidden_size)

        self.global_graph = GlobalGraph(hidden_size)

        self.decoder = Decoder(self, **decoder)

    def forward_encode_sub_graph(self,
                                 device,
                                 batch_size,
                                 agents_batch=None,
                                 agent_matrix=None,
                                 agent_matrix_slices=None,
                                 **kwargs) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        :param agents_batch: each value in list is vectors of all element (shape [-1, 128])
        :return: hidden states of all elements 
        """
        assert batch_size == 1, batch_size

        if agent_matrix is not None:
            agent_input_list_list = []
            for i in range(batch_size):
                input_list = []
                for j, each in enumerate(agent_matrix_slices[i]):
                    tensor = torch.tensor(agent_matrix[i][each], device=device, dtype=torch.float)
                    input_list.append(tensor)
                agent_input_list_list.append(input_list)

            if agents_batch is None:
                agents_batch = []
                for i in range(batch_size):
                    assert len(agent_input_list_list[i]) > 0
                    a, _ = self.point_level_sub_graph(agent_input_list_list[i])
                    agents_batch.append(a)
            else:
                for i in range(batch_size):
                    assert len(agent_input_list_list[i]) > 0
                    a, _ = self.point_level_sub_graph(agent_input_list_list[i])
                    assert len(agents_batch[i]) == len(a)
                    agents_batch[i] = agents_batch[i] + a
        else:
            assert agents_batch is not None

        element_states_batch = []
        for i in range(batch_size):
            agents = agents_batch[i]
            element_states_batch.append(agents)

        return element_states_batch, agents_batch

    # @profile
    def forward(self,
                mapping=None,
                labels=None,
                labels_is_valid=None,
                agents: List[Tensor] = None,
                agents_indices=None,
                device=None,
                **kwargs,
                ):
        import time
        global starttime
        starttime = time.time()

        mapping = mapping
        if 'work_dir' in mapping and np.random.randint(20) == 0:
            print(f'work_dir {mapping["work_dir"]}')
        if np.random.randint(0, 50) == 0:
            print('index', mapping[0]['index'], device)
        if agents is not None:
            agents = [each[:, :128] for each in agents]

        batch_size = len(agents)

        element_states_batch, agents = self.forward_encode_sub_graph(device, batch_size, agents_batch=agents, **kwargs)

        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask, mapping)

        return self.decoder(mapping, batch_size, None, inputs, inputs_lengths, hidden_states, device,
                            labels, labels_is_valid, agents_indices, agents=agents, **kwargs)
