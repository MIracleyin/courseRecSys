from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import ModelType as RecBoleModelType

class GeneralGraphRecommender(GeneralRecommender):
    """This is an abstract general graph recommender. All the general graph models should implement in this class.
    The base general graph recommender class provide the basic U-I graph dataset and parameters information.
    """
    type = RecBoleModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralGraphRecommender, self).__init__(config, dataset)
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)

class SequentialGraphRecommender(GeneralGraphRecommender):
    type = RecBoleModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SequentialGraphRecommender, self).__init__(config, dataset)
        self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = dataset.num(self.ITEM_ID)

