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


class BiGNNConv(MessagePassing):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
            output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_channels, out_channels):
        super(BiGNNConv, self).__init__(aggr='add')
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

class TPGNN(SequentialGraphRecommender):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(TPGNN, self).__init__(config, dataset)

        # load dataset info
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
        self.GNNlayers1 = torch.nn.ModuleList()
        for input_size, output_size in zip(self.hidden_size_list[:-1], self.hidden_size_list[1:]):
            self.GNNlayers1.append(BiGNNConv(input_size, output_size))
        self.GNNlayers2 = torch.nn.ModuleList()
        for input_size, output_size in zip(self.hidden_size_list[:-1], self.hidden_size_list[1:]):
            self.GNNlayers2.append(BiGNNConv(input_size, output_size))
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.conv1d = nn.Conv1d(self.embedding_size, self.embedding_size, kernel_size=self.sequential_len)  #
        self.pooling = nn.MaxPool1d(kernel_size=self.sequential_len)
        self.attlayer = AttLayer(self.embedding_size, 64)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=0, index=gather_index)
        return output_tensor.squeeze(1)

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

    def forward(self, item_seq, item_seq_len):
        if self.node_dropout == 0:
            edge_index, edge_weight = self.edge_index, self.edge_weight
            cedge_index, cedge_weight = self.c_edge_index, self.c_edge_weight
        else:
            edge_index, edge_weight = dropout_adj(edge_index=self.edge_index, edge_attr=self.edge_weight,
                                                  p=self.node_dropout)
            cedge_index, cedge_weight = dropout_adj(edge_index=self.c_edge_index, edge_attr=self.c_edge_weight,
                                                  p=self.node_dropout)
        #ui

        ui_embeddings = self.get_ego_embeddings()
        temp_embeddings_list = [ui_embeddings]
        for gnn in self.GNNlayers1:
            ui_embeddings = gnn(ui_embeddings, edge_index, edge_weight)
            ui_embeddings = nn.LeakyReLU(negative_slope=0.2)(ui_embeddings)
            ui_embeddings = nn.Dropout(self.message_dropout)(ui_embeddings)
            ui_embeddings = F.normalize(ui_embeddings, p=2, dim=1)
            temp_embeddings_list += [ui_embeddings]  # storage output embedding of each layer
        ui_embeddings = torch.stack(temp_embeddings_list, dim=1)
        ui_embeddings = torch.mean(ui_embeddings, dim=1)

        user_all_embeddings, uiitem_embeddings = torch.split(ui_embeddings, [self.n_users, self.n_items])

        iuc_embeddings = self.get_iuc_embeddings()
        temp_embeddings_list = [iuc_embeddings]
        for gnn in self.GNNlayers1 :
            iuc_embeddings = gnn(iuc_embeddings, cedge_index, cedge_weight)
            iuc_embeddings = nn.LeakyReLU(negative_slope=0.2)(iuc_embeddings)
            iuc_embeddings = nn.Dropout(self.message_dropout)(iuc_embeddings)
            iuc_embeddings = F.normalize(iuc_embeddings, p=2, dim=1)
            temp_embeddings_list += [iuc_embeddings]
        iuc_embeddings = torch.stack(temp_embeddings_list, dim=1)
        iuc_embeddings = torch.mean(iuc_embeddings, dim=1)

        item_all_embeddings, _, _ =  torch.split(iuc_embeddings, [self.n_items, self.n_users, self.n_cates])
        #
        # item_all_embeddings = torch.stack([uiitem_embeddings, iucitem_embeddings], dim=1)
        # item_all_embeddings = torch.mean(item_all_embeddings, dim=1)




        return user_all_embeddings, item_all_embeddings#, item_seq_emb_final

    def sampled_forward(self, batch_user, batch_pos, batch_neg, item_seq, item_seq_len):
        all_embeddings = self.get_ego_embeddings()
        batch = torch.concat([batch_user, batch_pos, batch_neg]).to('cpu')

        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            sampled_edge_index, sampled_edge_weight = self.sample_edges(batch, 30)

            all_embeddings = gnn(all_embeddings, sampled_edge_index, sampled_edge_weight)
            embeddings_list.append(all_embeddings)
        gnn_all_embeddings = torch.stack(embeddings_list, dim=1)
        gnn_all_embeddings = torch.mean(gnn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(gnn_all_embeddings, [self.n_users, self.n_items])

        item_seq_emb = self.item_embedding(item_seq)
        # item_seq_emb = self.emb_dropout(item_seq_emb)
        item_seq_emb = item_seq_emb.permute(0, 2, 1)
        item_seq_emb = self.conv1d(item_seq_emb)
        item_seq_emb = self.pooling(item_seq_emb)
        item_seq_emb = item_seq_emb.permute(0, 2, 1)  # changback dim

        item_seq_emb_final = self.gather_indexes(item_seq_emb, item_seq_len - 1)

        return user_all_embeddings, item_all_embeddings, item_seq_emb_final

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
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        if not self.graph_sample:
            user_all_embeddings, item_all_embeddings = self.forward(item_seq, item_seq_len)
        else:
            user_all_embeddings, item_all_embeddings, item_seq_emb = self.sampled_forward(user, pos_item, neg_item,
                                                                                          item_seq, item_seq_len)

        # user_all_embeddings, item_all_embeddings, item_seq_emb = self.forward(item_seq, item_seq_len)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # pos_embeddings = torch.stack([pos_embeddings, item_seq_emb], dim=1)
        # pos_embeddings = torch.mean(pos_embeddings, dim=1)

        # att_signal = self.attlayer(pos_embeddings.unsqueeze(2))

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

        user_all_embeddings, item_all_embeddings= self.forward(item_seq, item_seq_len)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        # i_embeddings = torch.stack([i_embeddings, item_seq_emb], dim=1)
        # i_embeddings = torch.mean(i_embeddings, dim=1)

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(item_seq, item_seq_len)
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]
        # gather_index = torch.arange(0, self.n_items).to(self.device)

        # self.item_seq_emb = self.gather_indexes(self.item_seq_emb.unsqueeze(1), gather_index)
        #
        # self.restore_item_e = torch.stack([self.restore_item_e, self.item_seq_emb], dim=1)
        # self.restore_item_e = torch.mean(self.restore_item_e, dim=1)

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
