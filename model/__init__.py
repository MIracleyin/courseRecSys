from .pop import Pop
from .bpr import BPR
from .gru4rec import GRU4Rec
from .pinsage import PinSage
from .lightgcn import LightGCN
from .ngcf import NGCF
from .tpgnn import TPGNN


model_name_map = {
    'Pop': Pop,
    'BPR': BPR,
    'GRU4Rec': GRU4Rec,
    'PinSage': PinSage,
    'LightGCN': LightGCN,
    'NGCF': NGCF,
    'TPGNN': TPGNN,

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