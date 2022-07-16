from .pop import Pop
from .bpr import BPR
from .gru4rec import GRU4Rec

model_name_map = {
    'Pop': Pop,
    'BPR': BPR,
    'GRU4Rec': GRU4Rec
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