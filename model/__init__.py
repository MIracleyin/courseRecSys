from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import ModelType as RecBoleModelType

from .pop import Pop
from .bpr import BPR
from .gru4rec import GRU4Rec
from .pinsage import PinSage

class GeneralGraphRecommender(GeneralRecommender):
    """This is an abstract general graph recommender. All the general graph models should implement in this class.
    The base general graph recommender class provide the basic U-I graph dataset and parameters information.
    """
    type = RecBoleModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralGraphRecommender, self).__init__(config, dataset)
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)


model_name_map = {
    'Pop': Pop,
    'BPR': BPR,
    'GRU4Rec': GRU4Rec,
    'PinSage': PinSage
    # 'BPR-T': ExtendedBPR,
    # 'CFA': CFautoencoder,
    # 'DSPR': DeepSimPersionalRec,
    # 'LGCN' : LightGCN,
    # 'LAGCF': LAGCF,
    # 'NGCF': NGCF,
    # 'NGCFT': NGCF_T,
    # 'SGL': SGL,
    # 'TGCN': TGCN
}