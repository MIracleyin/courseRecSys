# courseRecSys
This repo is benchmark for mooc recommendation systems

## Dataset
ln -s xxx/mooc_data ./dataset/

## Baseline
### Pop
test result: OrderedDict([('recall@10', 0.1421), ('mrr@10', 0.036), ('ndcg@10', 0.0602), ('hit@10', 0.1444), ('precision@10', 0.0145)])

### BPR
test result: OrderedDict([('recall@10', 0.4004), ('mrr@10', 0.2424), ('ndcg@10', 0.2784), ('hit@10', 0.4043), ('precision@10', 0.041)])

### GRU4Rec
test result: OrderedDict([('recall@10', 0.4535), ('mrr@10', 0.2505), ('ndcg@10', 0.2983), ('hit@10', 0.4535), ('precision@10', 0.0454)])

### pinsage