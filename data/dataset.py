import torch
from tqdm import tqdm
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from recbole.data.dataset import Dataset as RecBoleDatase
from recbole.data.dataset import SequentialDataset as RecBoleSeqDataset


class GeneralGraphDataset(RecBoleDatase):
    def __init__(self, config):
        super(GeneralGraphDataset, self).__init__(config)

    @property
    def node_num(self):
        """Get the number of different tokens of ``self.uid_field``.

        Returns:
            int: Number of different tokens of ``self.uid_field``.
        """
        self._check_field('uid_field')
        self._check_field('iid_field')
        return self.num(self.uid_field) + self.num(self.iid_field)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = self.inter_feat[self.uid_field]
        col = self.inter_feat[self.iid_field] + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def get_bipartite_inter_mat(self, row='user', row_norm=True):
        r"""Get the row-normalized bipartite interaction matrix of users and items.
                """
        if row == 'user':
            row_field, col_field = self.uid_field, self.iid_field
        else:
            row_field, col_field = self.iid_field, self.uid_field

        row = self.inter_feat[row_field]
        col = self.inter_feat[col_field]
        edge_index = torch.stack([row, col])

        if row_norm:  # only row normalize
            deg = degree(edge_index[0], self.num(row_field))
            norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)  # if deg == 0, norm = 1 else 1/deg
            edge_weight = norm_deg[edge_index[0]]
        else:
            row_deg = degree(edge_index[0], self.num(row_field))
            col_deg = degree(edge_index[1], self.num(col_field))

            row_norm_deg = 1. / torch.sqrt(torch.where(row_deg == 0, torch.ones([1]), row_deg))
            col_norm_deg = 1. / torch.sqrt(torch.where(col_deg == 0, torch.ones([1]), col_deg))

            edge_weight = row_norm_deg[edge_index[0]] * col_norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def get_neighbor_sample_mat(self, edge_index, edge_weight):
        edge_index = edge_index.to('cpu')
        edge_weight = edge_weight.to('cpu')

        adj_mat = SparseTensor(row=edge_index[0], col=edge_index[1],
                               value=edge_weight,
                               sparse_sizes=(self.node_num, self.node_num)).t()
        return adj_mat


class SequentialGraphDataset(RecBoleSeqDataset):
    def __init__(self, config):
        super(SequentialGraphDataset, self).__init__(config)

    @property
    def node_num(self):
        """Get the number of different tokens of ``self.uid_field``.

        Returns:
            int: Number of different tokens of ``self.uid_field``.
        """
        self._check_field('uid_field')
        self._check_field('iid_field')
        return self.num(self.uid_field) + self.num(self.iid_field)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = self.inter_feat[self.uid_field]
        col = self.inter_feat[self.iid_field] + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def get_bipartite_inter_mat(self, row='user', row_norm=True):
        r"""Get the row-normalized bipartite interaction matrix of users and items.
                """
        if row == 'user':
            row_field, col_field = self.uid_field, self.iid_field
        else:
            row_field, col_field = self.iid_field, self.uid_field

        row = self.inter_feat[row_field]
        col = self.inter_feat[col_field]
        edge_index = torch.stack([row, col])

        if row_norm:  # only row normalize
            deg = degree(edge_index[0], self.num(row_field))
            norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)  # if deg == 0, norm = 1 else 1/deg
            edge_weight = norm_deg[edge_index[0]]
        else:
            row_deg = degree(edge_index[0], self.num(row_field))
            col_deg = degree(edge_index[1], self.num(col_field))

            row_norm_deg = 1. / torch.sqrt(torch.where(row_deg == 0, torch.ones([1]), row_deg))
            col_norm_deg = 1. / torch.sqrt(torch.where(col_deg == 0, torch.ones([1]), col_deg))

            edge_weight = row_norm_deg[edge_index[0]] * col_norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def get_neighbor_sample_mat(self, edge_index, edge_weight):
        edge_index = edge_index.to('cpu')
        edge_weight = edge_weight.to('cpu')

        adj_mat = SparseTensor(row=edge_index[0], col=edge_index[1],
                               value=edge_weight,
                               sparse_sizes=(self.node_num, self.node_num)).t()
        return adj_mat

    def session_graph_construction(self):
        # Default session graph dataset follows the graph construction operator like SR-GNN.
        self.logger.info('Constructing session graphs.')
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]
        x = []
        edge_index = []
        alias_inputs = []

        for i, seq in enumerate(tqdm(list(torch.chunk(item_seq, item_seq.shape[0])))):
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)
            # No repeat click
            edge = torch.stack([alias_seq[:-1], alias_seq[1:]]).unique(dim=-1)
            edge_index.append(edge)

        self.inter_feat.interaction['graph_idx'] = torch.arange(item_seq.shape[0])
        self.graph_objs = {
            'x': x,
            'edge_index': edge_index,
            'alias_inputs': alias_inputs
        }

    def build(self):
        datasets = super().build()
        for dataset in datasets:
            dataset.session_graph_construction()
        return datasets
