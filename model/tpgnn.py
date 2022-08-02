import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.init import xavier_normal_initialization

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType, ModelType

from model.generalgraphrecommender import GeneralGraphRecommender, SequentialGraphRecommender
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation

from torch_sparse import SparseTensor
from torch import Tensor
from typing import Optional


class BiGNNConv(MessagePassing):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
            output = (L+I)EW_1 + LE \otimes EW_2
    """
    def __init__(self, in_channels, out_channels, aggr='add'):
        # aggr = CNNAggregation(64,64,5)
        super(BiGNNConv, self).__init__(aggr=aggr)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin1 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.lin2 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_weight):
        x_prop = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x_trans = self.lin1(x_prop + x)
        x_inter = self.lin2(torch.mul(x_prop, x))
        return x_trans + x_inter

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class CNNAggregation(Aggregation):
    def __init__(self, in_channels, out_channels, kernal_size):
        super(CNNAggregation, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernal_size = kernal_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernal_size)
        self.reset_parameters()
        self.pooling = nn.MaxPool1d(kernel_size=kernal_size)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        fill_value = x.min().item() - 1
        sortedIndex, _ = torch.sort(index)
        x, _ = self.to_dense_batch(x, sortedIndex, ptr, dim_size, dim, fill_value=fill_value)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pooling(x)
        x = x.permute(0, 2, 1)
        x = torch.mean(x, dim=1)
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class AttLayer(nn.Module):
    """Calculate the attention signal(weight) according the input tensor.

    Args:
        infeatures (torch.FloatTensor): A 3D input tensor with shape of[batch_size, M, embed_dim].

    Returns:
        torch.FloatTensor: Attention weight of input. shape of [batch_size, M].
    """

    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_signal = self.w(infeatures)  # [batch_size, M, att_dim]
        att_signal = F.relu(att_signal)  # [batch_size, M, att_dim]

        att_signal = torch.mul(att_signal, self.h)  # [batch_size, M, att_dim]
        att_signal = torch.sum(att_signal, dim=2)  # [batch_size, M]
        att_signal = F.softmax(att_signal, dim=1)  # [batch_size, M]

        return att_signal


class TPGNN(GeneralGraphRecommender):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(TPGNN, self).__init__(config, dataset)
        # load dataset info
        self.CATE_ID = config['CATE_FIELD']
        self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_cates = dataset.num(self.CATE_ID)

        self.c_edge_index, self.c_edge_weight = dataset.get_c_norm_adj_mat()
        self.c_edge_index, self.c_edge_weight = self.c_edge_index.to(self.device), self.c_edge_weight.to(self.device)
        self.sample_edge_index = dataset.get_neighbor_sample_mat(self.edge_index, self.edge_weight)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size_list = config['hidden_size_list']
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.num_layers = len(self.hidden_size_list)
        self.node_dropout = config['node_dropout']
        self.message_dropout = config['message_dropout']
        self.reg_weight = config['reg_weight']
        self.sequential_len = config['sequential_len']
        self.propagation_dept = config['propagation_depth']
        self.dropout_prob = config['dropout_prob']
        self.graph_sample = config['graph_sample']
        assert self.num_layers == self.propagation_dept

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.cate_embedding = nn.Embedding(self.n_cates, self.embedding_size)
        # user embeddings
        self.GNNlayersUser = torch.nn.ModuleList()
        for input_size, output_size in zip(self.hidden_size_list[:-1], self.hidden_size_list[1:]):
            self.GNNlayersUser.append(BiGNNConv(input_size, output_size))
        self.layersUserCNN = torch.nn.ModuleList()
        for input_size, output_size in zip(self.hidden_size_list[:-1], self.hidden_size_list[1:]):
            aggr = CNNAggregation(input_size, output_size, kernal_size=self.sequential_len)
            self.layersUserCNN.append(BiGNNConv(input_size, output_size, aggr=aggr))
        self.attlayer = AttLayer(self.embedding_size, 64)
        # item embeddings
        self.GNNlayersItem = torch.nn.ModuleList()
        for input_size, output_size in zip(self.hidden_size_list[:-1], self.hidden_size_list[1:]):
            self.GNNlayersItem.append(BiGNNConv(input_size, output_size))
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_iuc_embeddings(self):
        item_embeddings = self.item_embedding.weight
        user_embeddings = self.user_embedding.weight
        cate_embeddings = self.cate_embedding.weight
        iuc_embeddings = torch.cat([item_embeddings, user_embeddings, cate_embeddings], dim=0)
        return iuc_embeddings

    def forward(self, batch_user, batch_pos=None, batch_neg=None):
        if self.node_dropout == 0:
            edge_index, edge_weight = self.edge_index, self.edge_weight
            cedge_index, cedge_weight = self.c_edge_index, self.c_edge_weight
        else:
            edge_index, edge_weight = dropout_adj(edge_index=self.edge_index, edge_attr=self.edge_weight,
                                                  p=self.node_dropout)
            cedge_index, cedge_weight = dropout_adj(edge_index=self.c_edge_index, edge_attr=self.c_edge_weight,
                                                    p=self.node_dropout)
        # ui
        ui_embeddings = self.get_ego_embeddings()
        temp_embeddings_list = [ui_embeddings]
        for gnn, gnncnn in zip(self.GNNlayersUser, self.layersUserCNN):
            # sample
            # if batch_neg == None and batch_pos == None: # inference
            #     batch = torch.concat([batch_user]).to('cpu')
            # else: # train
            #    batch = torch.concat([batch_user, batch_pos, batch_neg]).to('cpu')
            # sampled_edge_index, sampled_edge_weight = self.sample_edges(batch, 30)
            ui_embeddings = gnn(ui_embeddings, edge_index, edge_weight)
            # ui_embeddings = nn.LeakyReLU(negative_slope=0.2)(ui_embeddings)
            ui_embeddings = nn.Dropout(self.message_dropout)(ui_embeddings)
            # ui_embeddings = F.normalize(ui_embeddings, p=2, dim=1)
            # ui_embeddings = gnncnn(ui_embeddings, sampled_edge_index, sampled_edge_weight)  # get back sort imformation
            temp_embeddings_list += [ui_embeddings]  # storage output embedding of each layer
        ui_embeddings = torch.stack(temp_embeddings_list, dim=1)
        ui_embeddings = torch.mean(ui_embeddings, dim=1)

        user_all_embeddings, uiitem_embeddings = torch.split(ui_embeddings, [self.n_users, self.n_items])

        # user cnn
        # ui_embeddings = self.conv(ui_embeddings, edge_index, edge_weight)

        iuc_embeddings = self.get_iuc_embeddings()
        temp_embeddings_list = [iuc_embeddings]
        for gnn in self.GNNlayersUser:
            iuc_embeddings = gnn(iuc_embeddings, cedge_index, cedge_weight)
            # iuc_embeddings = nn.LeakyReLU(negative_slope=0.2)(iuc_embeddings)
            iuc_embeddings = nn.Dropout(self.message_dropout)(iuc_embeddings)
            # iuc_embeddings = F.normalize(iuc_embeddings, p=2, dim=1)
            temp_embeddings_list += [iuc_embeddings]

        iuc_embeddings = torch.stack(temp_embeddings_list, dim=1)
        iuc_embeddings = torch.mean(iuc_embeddings, dim=1)

        item_all_embeddings, _, _ = torch.split(iuc_embeddings, [self.n_items, self.n_users, self.n_cates])
        #
        # item_all_embeddings = torch.stack([uiitem_embeddings, iucitem_embeddings], dim=1)
        # item_all_embeddings = torch.mean(item_all_embeddings, dim=1)

        return user_all_embeddings, item_all_embeddings  # , item_seq_emb_final

    def sample_edges(self, batch, num_neighbor, replace=False):
        adj_t, n_id = self.sample_edge_index.sample_adj(batch, num_neighbor, replace=replace)
        row, col, data = adj_t.coo()
        edge_index = torch.stack([col, row], dim=0).to(self.device)
        edge_weight = data.to(self.device)
        return edge_index, edge_weight

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(user, pos_item, neg_item)

        # user_all_embeddings, item_all_embeddings, item_seq_emb = self.forward(item_seq, item_seq_len)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss

        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)  # L2 regularization of embeddings

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        user_all_embeddings, item_all_embeddings = self.forward(item_seq, item_seq_len)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        # i_embeddings = torch.stack([i_embeddings, item_seq_emb], dim=1)
        # i_embeddings = torch.mean(i_embeddings, dim=1)

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(user)
        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
