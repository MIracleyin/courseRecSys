from .pop import Pop
from .bpr import BPR
from .gru4rec import GRU4Rec
from .pinsage import PinSage
from .ngcf import NGCF
from .tpgnn import TPGNN
from .lightgcn import LightGCN
from .sgl import SGL

model_name_map = {
    'Pop': Pop,
    'BPR': BPR,
    'GRU4Rec': GRU4Rec,
    'PinSage': PinSage,
    'LightGCN': LightGCN,
    'NGCF': NGCF,
    'TPGNN': TPGNN,
    'SGL': SGL
}