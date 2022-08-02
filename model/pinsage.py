import torch
import torch.nn as nn
import torch.nn.functional as F
# import dgl.nn.pytorch as dglnn
# import dgl.function as fn


from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
from model.generalgraphrecommender import GeneralGraphRecommender


# class WeightedSAGEConv(nn.Module):
#     def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
#         super(WeightedSAGEConv, self).__init__()
#         self.act = act
#         self.Q = nn.Linear(input_dims, hidden_dims)
#         self.W = nn.Linear(input_dims + hidden_dims, output_dims)
#         self.reset_parameters()
#         self.dropout = nn.Dropout(0.5)
#
#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_uniform_(self.Q.weight, gain=gain)
#         nn.init.xavier_uniform_(self.W.weight, gain=gain)
#         nn.init.constant_(self.Q.bias, 0)
#         nn.init.constant_(self.W.bias, 0)
#
#     def forward(self, g, h, weights):
#         """
#         g : graph
#         h : node features
#         weights : scalar edge weights
#         """
#         h_src, h_dst = h
#         with g.local_scope():
#             g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
#             g.edata['w'] = weights.float()
#             g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
#             g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
#             n = g.dstdata['n']
#             ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
#             z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
#             z_norm = z.norm(2, 1, keepdim=True)
#             z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
#             z = z / z_norm
#             return z
#
# class SAGENet(nn.Module):
#     def __init__(self, hidden_dims, n_layers):
#         """
#         g : DGLHeteroGraph
#             The user-item interaction graph.
#             This is only for finding the range of categorical variables.
#         item_textsets : torchtext.data.Dataset
#             The textual features of each item node.
#         """
#         super().__init__()
#         self.convs = nn.ModuleList()
#         for _ in range(n_layers):
#             self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims))
#
#     def forward(self, blocks, h):
#         for layer, block in zip(self.convs, blocks):
#             h_dst = h[:block.num_nodes('DST/' + block.ntypes[0])]
#             h = layer(block, (h, h_dst), block.edata['weights'])
#         return h

class SAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGENet, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels, addr='add'))

    def forward(self, x, edge_index):
        # for i, (edge_index, _, size) in enumerate(adjs):
        #     x_target = x[:size[1]]
        #     x = self.convs[i]((x, x_target), edge_index)
        #     if i != self.num_layers - 1:
        #         x = x.relu()
        #         x = F.dropout(x, p=0.5, training=self.training)
        # return x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        self.dataset
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class PinSage(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(PinSage, self).__init__(config, dataset)

        # load dataset info
        self.sample_edge_index = dataset.get_neighbor_sample_mat(self.edge_index, self.edge_weight)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_dims = config['hidden_size']
        self.num_layers = config['num_layers']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization
        self.graph_sample = config['graph_sample']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.sage_conv = SAGENet(self.embedding_size, self.hidden_dims, self.num_layers)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_neighbor_sample(self):
        return

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.num_layers):
            all_embeddings = self.sage_conv(all_embeddings, self.edge_index)
            embeddings_list.append(all_embeddings)
        sage_all_embeddings = torch.stack(embeddings_list, dim=1)
        sage_all_embeddings = torch.mean(sage_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(sage_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def sampled_forward(self, batch_user, batch_pos, batch_neg):
        all_embeddings = self.get_ego_embeddings()
        batch = torch.concat([batch_user, batch_pos, batch_neg]).to('cpu')

        embeddings_list = [all_embeddings]
        for layer_idx in range(self.num_layers):
            sampled_edge_index = self.sample_edges(batch, 30)
            all_embeddings = self.sage_conv(all_embeddings, sampled_edge_index)
            embeddings_list.append(all_embeddings)
        sage_all_embeddings = torch.stack(embeddings_list, dim=1)
        sage_all_embeddings = torch.mean(sage_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(sage_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def sample_edges(self, batch, num_neighbor, replace=False):
        adj_t, n_id = self.sample_edge_index.sample_adj(batch, num_neighbor, replace=replace)
        row, col, _ = adj_t.coo()
        edge_index = torch.stack([col, row], dim=0).to(self.device)
        return edge_index

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        #
        if not self.graph_sample:
            user_all_embeddings, item_all_embeddings = self.forward()
        else:
            user_all_embeddings, item_all_embeddings = self.sampled_forward(user, pos_item, neg_item)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
