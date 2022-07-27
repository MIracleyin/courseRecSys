# courseRecSys
This repo is benchmark for mooc recommendation systems

## Dataset
ln -s xxx/mooc_data ./dataset/

## Baseline
### Pop
test result: OrderedDict([('recall@5', 0.1021), ('recall@10', 0.1699), ('recall@15', 0.2245), ('recall@20', 0.2862), ('ndcg@5', 0.0677), ('ndcg@10', 0.0898), ('ndcg@15', 0.1042), ('ndcg@20', 0.1187)])

### BPR
test result: OrderedDict([('recall@5', 0.3134), ('recall@10', 0.4015), ('recall@15', 0.4602), ('recall@20', 0.502), ('ndcg@5', 0.2516), ('ndcg@10', 0.2799), ('ndcg@15', 0.2956), ('ndcg@20', 0.3055)])

### GRU4Rec
MAX_ITEM_LIST_LENGTH: 50
test result: OrderedDict([('recall@5', 0.3629), ('recall@10', 0.4547), ('recall@15', 0.5136), ('recall@20', 0.5566), ('ndcg@5', 0.2741), ('ndcg@10', 0.3037), ('ndcg@15', 0.3193), ('ndcg@20', 0.3294)])
MAX_ITEM_LIST_LENGTH: 7 reference to TP-GNN
test result: OrderedDict([('recall@5', 0.3622), ('recall@10', 0.4594), ('recall@15', 0.5281), ('recall@20', 0.5723), ('ndcg@5', 0.2657), ('ndcg@10', 0.2971), ('ndcg@15', 0.3153), ('ndcg@20', 0.3257)])
loss_type: 'BPR' not suit for Sequence Rec
test result: OrderedDict([('recall@5', 0.288), ('recall@10', 0.4129), ('recall@15', 0.4797), ('recall@20', 0.5289), ('ndcg@5', 0.1779), ('ndcg@10', 0.2182), ('ndcg@15', 0.2359), ('ndcg@20', 0.2475)])

### pinsage
num_layers: 1
test result: OrderedDict([('recall@5', 0.2741), ('recall@10', 0.3528), ('recall@15', 0.4058), ('recall@20', 0.447), ('ndcg@5', 0.2125), ('ndcg@10', 0.238), ('ndcg@15', 0.2521), ('ndcg@20', 0.2619)])
num_layers: 2
test result: OrderedDict([('recall@5', 0.2598), ('recall@10', 0.3479), ('recall@15', 0.4055), ('recall@20', 0.4447), ('ndcg@5', 0.2003), ('ndcg@10', 0.2287), ('ndcg@15', 0.2441), ('ndcg@20', 0.2534)])
num_layers: 3
test result: OrderedDict([('recall@5', 0.2675), ('recall@10', 0.3585), ('recall@15', 0.4172), ('recall@20', 0.4601), ('ndcg@5', 0.2121), ('ndcg@10', 0.2413), ('ndcg@15', 0.2569), ('ndcg@20', 0.2671)])
num_layers: 4 over-smooth
test result: OrderedDict([('recall@5', 0.2399), ('recall@10', 0.3302), ('recall@15', 0.3878), ('recall@20', 0.4327), ('ndcg@5', 0.1902), ('ndcg@10', 0.2193), ('ndcg@15', 0.2348), ('ndcg@20', 0.2454)])
num_neighbor: 30
test result: OrderedDict([('recall@5', 0.1153), ('recall@10', 0.185), ('recall@15', 0.2221), ('recall@20', 0.2788), ('ndcg@5', 0.0742), ('ndcg@10', 0.0961), ('ndcg@15', 0.1061), ('ndcg@20', 0.1195)])
num_neighbor: 300
test result: OrderedDict([('recall@5', 0.1233), ('recall@10', 0.1873), ('recall@15', 0.2192), ('recall@20', 0.2747), ('ndcg@5', 0.0757), ('ndcg@10', 0.0953), ('ndcg@15', 0.1039), ('ndcg@20', 0.117)])
num_neighbor: 3000
test result: OrderedDict([('recall@5', 0.106), ('recall@10', 0.1897), ('recall@15', 0.2235), ('recall@20', 0.2899), ('ndcg@5', 0.0711), ('ndcg@10', 0.0976), ('ndcg@15', 0.1067), ('ndcg@20', 0.1224)])

### LightGCN
num_layers: 2
test result: OrderedDict([('recall@5', 0.3565), ('recall@10', 0.4579), ('recall@15', 0.5215), ('recall@20', 0.5647), ('ndcg@5', 0.2812), ('ndcg@10', 0.314), ('ndcg@15', 0.331), ('ndcg@20', 0.3413)])
num_layers: 3
test result: OrderedDict([('recall@5', 0.3581), ('recall@10', 0.4604), ('recall@15', 0.5209), ('recall@20', 0.563), ('ndcg@5', 0.287), ('ndcg@10', 0.3201), ('ndcg@15', 0.3362), ('ndcg@20', 0.3462)])
num_layers: 4
test result: OrderedDict([('recall@5', 0.3678), ('recall@10', 0.4643), ('recall@15', 0.5245), ('recall@20', 0.5659), ('ndcg@5', 0.2912), ('ndcg@10', 0.3225), ('ndcg@15', 0.3385), ('ndcg@20', 0.3484)])
num_layers: 5
test result: OrderedDict([('recall@5', 0.3704), ('recall@10', 0.4712), ('recall@15', 0.5288), ('recall@20', 0.5674), ('ndcg@5', 0.295), ('ndcg@10', 0.3276), ('ndcg@15', 0.343), ('ndcg@20', 0.3522)])
num_layers: 6 over-smooth
test result: OrderedDict([('recall@5', 0.3635), ('recall@10', 0.4608), ('recall@15', 0.5193), ('recall@20', 0.563), ('ndcg@5', 0.2848), ('ndcg@10', 0.3163), ('ndcg@15', 0.3318), ('ndcg@20', 0.3422)])

### TP-GNN
test result: OrderedDict([('recall@5', 0.3291), ('recall@10', 0.4203), ('recall@15', 0.4811), ('recall@20', 0.5261), ('ndcg@5', 0.2601), ('ndcg@10', 0.2895), ('ndcg@15', 0.3057), ('ndcg@20', 0.3164)])
