# courseRecSys
This repo is benchmark for mooc recommendation systems

## Dataset
ln -s xxx/mooc_data ./dataset/

# RO
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


### TP-GNN
test result: OrderedDict([('recall@5', 0.3291), ('recall@10', 0.4203), ('recall@15', 0.4811), ('recall@20', 0.5261), ('ndcg@5', 0.2601), ('ndcg@10', 0.2895), ('ndcg@15', 0.3057), ('ndcg@20', 0.3164)])
test result: OrderedDict([('recall@5', 0.0894), ('recall@10', 0.1845), ('recall@15', 0.2587), ('recall@20', 0.2971), ('ndcg@5', 0.057), ('ndcg@10', 0.0864), ('ndcg@15', 0.1061), ('ndcg@20', 0.1151)])
test result: OrderedDict([('recall@5', 0.0922), ('recall@10', 0.164), ('recall@15', 0.2489), ('recall@20', 0.2887), ('ndcg@5', 0.0583), ('ndcg@10', 0.0813), ('ndcg@15', 0.1038), ('ndcg@20', 0.1132)])
test result: OrderedDict([('recall@5', 0.0812), ('recall@10', 0.1362), ('recall@15', 0.1708), ('recall@20', 0.2159), ('ndcg@5', 0.0516), ('ndcg@10', 0.0694), ('ndcg@15', 0.0786), ('ndcg@20', 0.0892)])
test result: OrderedDict([('recall@5', 0.101), ('recall@10', 0.2249), ('recall@15', 0.2822), ('recall@20', 0.3248), ('ndcg@5', 0.0596), ('ndcg@10', 0.0982), ('ndcg@15', 0.1134), ('ndcg@20', 0.1234)])

#### but have over fitting
seq_len 7
best valid : OrderedDict([('recall@5', 0.1513), ('recall@10', 0.2784), ('recall@15', 0.3328), ('recall@20', 0.3745), ('ndcg@5', 0.0884), ('ndcg@10', 0.129), ('ndcg@15', 0.1434), ('ndcg@20', 0.1532)])
test result: OrderedDict([('recall@5', 0.0989), ('recall@10', 0.2084), ('recall@15', 0.27), ('recall@20', 0.3158), ('ndcg@5', 0.0572), ('ndcg@10', 0.0917), ('ndcg@15', 0.108), ('ndcg@20', 0.1189)])
seq_len 5
best valid : OrderedDict([('recall@5', 0.1556), ('recall@10', 0.2834), ('recall@15', 0.3388), ('recall@20', 0.3815), ('ndcg@5', 0.091), ('ndcg@10', 0.1321), ('ndcg@15', 0.1469), ('ndcg@20', 0.1569)])
test result: OrderedDict([('recall@5', 0.102), ('recall@10', 0.2125), ('recall@15', 0.2758), ('recall@20', 0.3169), ('ndcg@5', 0.0595), ('ndcg@10', 0.0943), ('ndcg@15', 0.1111), ('ndcg@20', 0.1208)])
seq_len 3
best valid : OrderedDict([('recall@5', 0.1211), ('recall@10', 0.2672), ('recall@15', 0.3298), ('recall@20', 0.3747), ('ndcg@5', 0.0681), ('ndcg@10', 0.1141), ('ndcg@15', 0.1307), ('ndcg@20', 0.1413)])
test result: OrderedDict([('recall@5', 0.1022), ('recall@10', 0.2164), ('recall@15', 0.275), ('recall@20', 0.3186), ('ndcg@5', 0.0589), ('ndcg@10', 0.0947), ('ndcg@15', 0.1103), ('ndcg@20', 0.1206)])
MAX_ITEM_LIST_LENGTH: 7 seq_len 2

test result: OrderedDict([('recall@5', 0.1034), ('recall@10', 0.2067), ('recall@15', 0.2756), ('recall@20', 0.3139), ('ndcg@5', 0.0627), ('ndcg@10', 0.095), ('ndcg@15', 0.1134), ('ndcg@20', 0.1225)])



### NGCF
test result: OrderedDict([('recall@5', 0.3195), ('recall@10', 0.4187), ('recall@15', 0.4859), ('recall@20', 0.5303), ('ndcg@5', 0.2504), ('ndcg@10', 0.2826), ('ndcg@15', 0.3005), ('ndcg@20', 0.3111)])


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

### SGL
num_layers: 3
test result: OrderedDict([('recall@5', 0.3675), ('recall@10', 0.4637), ('recall@15', 0.5204), ('recall@20', 0.5659), ('ndcg@5', 0.2954), ('ndcg@10', 0.3264), ('ndcg@15', 0.3416), ('ndcg@20', 0.3524)])
num_layers: 4
test result: OrderedDict([('recall@5', 0.37), ('recall@10', 0.4653), ('recall@15', 0.5209), ('recall@20', 0.5629), ('ndcg@5', 0.2971), ('ndcg@10', 0.3281), ('ndcg@15', 0.3428), ('ndcg@20', 0.3528)])
ssl_tau: 0.45
test result: OrderedDict([('recall@5', 0.3735), ('recall@10', 0.4661), ('recall@15', 0.5215), ('recall@20', 0.5631), ('ndcg@5', 0.3001), ('ndcg@10', 0.33), ('ndcg@15', 0.3448), ('ndcg@20', 0.3546)])
ssl_tau: 0.55
test result: OrderedDict([('recall@5', 0.3728), ('recall@10', 0.4645), ('recall@15', 0.52), ('recall@20', 0.5614), ('ndcg@5', 0.3027), ('ndcg@10', 0.3323), ('ndcg@15', 0.3471), ('ndcg@20', 0.3569)])
test result: OrderedDict([('recall@5', 0.3659), ('recall@10', 0.4649), ('recall@15', 0.5202), ('recall@20', 0.5622), ('ndcg@5', 0.2934), ('ndcg@10', 0.3255), ('ndcg@15', 0.3402), ('ndcg@20', 0.3502)])

# Change to TO eval
## Pop
test result: OrderedDict([('recall@5', 0.0418), ('recall@10', 0.0896), ('recall@15', 0.1341), ('recall@20', 0.1818), ('ndcg@5', 0.022), ('ndcg@10', 0.0373), ('ndcg@15', 0.049), ('ndcg@20', 0.0601)])

## BPR
best valid : OrderedDict([('recall@5', 0.2972), ('recall@10', 0.4019), ('recall@15', 0.4588), ('recall@20', 0.5027), ('ndcg@5', 0.2052), ('ndcg@10', 0.2391), ('ndcg@15', 0.2542), ('ndcg@20', 0.2647)])
test result: OrderedDict([('recall@5', 0.2515), ('recall@10', 0.329), ('recall@15', 0.386), ('recall@20', 0.4251), ('ndcg@5', 0.1756), ('ndcg@10', 0.2008), ('ndcg@15', 0.216), ('ndcg@20', 0.2252)])

## Gru4Rec
best valid : OrderedDict([('recall@5', 0.3231), ('recall@10', 0.4243), ('recall@15', 0.4882), ('recall@20', 0.5385), ('ndcg@5', 0.1966), ('ndcg@10', 0.2293), ('ndcg@15', 0.2462), ('ndcg@20', 0.258)])
test result: OrderedDict([('recall@5', 0.2984), ('recall@10', 0.4094), ('recall@15', 0.4765), ('recall@20', 0.5274), ('ndcg@5', 0.1858), ('ndcg@10', 0.2216), ('ndcg@15', 0.2393), ('ndcg@20', 0.2514)])

## PinSage
best valid : OrderedDict([('recall@5', 0.0735), ('recall@10', 0.1433), ('recall@15', 0.1637), ('recall@20', 0.1995), ('ndcg@5', 0.053), ('ndcg@10', 0.075), ('ndcg@15', 0.0803), ('ndcg@20', 0.0888)])
test result: OrderedDict([('recall@5', 0.0649), ('recall@10', 0.103), ('recall@15', 0.1331), ('recall@20', 0.1772), ('ndcg@5', 0.047), ('ndcg@10', 0.059), ('ndcg@15', 0.0667), ('ndcg@20', 0.0772)])


## TPGNN
best valid : OrderedDict([('recall@5', 0.1062), ('recall@10', 0.2554), ('recall@15', 0.308), ('recall@20', 0.3468), ('ndcg@5', 0.0631), ('ndcg@10', 0.109), ('ndcg@15', 0.1229), ('ndcg@20', 0.1321)])
test result: OrderedDict([('recall@5', 0.0835), ('recall@10', 0.2022), ('recall@15', 0.2544), ('recall@20', 0.2899), ('ndcg@5', 0.0504), ('ndcg@10', 0.0873), ('ndcg@15', 0.1011), ('ndcg@20', 0.1095)])
best valid : OrderedDict([('recall@5', 0.1509), ('recall@10', 0.327), ('recall@15', 0.3975), ('recall@20', 0.4446), ('ndcg@5', 0.092), ('ndcg@10', 0.1463), ('ndcg@15', 0.1651), ('ndcg@20', 0.1762)])
test result: OrderedDict([('recall@5', 0.1435), ('recall@10', 0.2908), ('recall@15', 0.3466), ('recall@20', 0.3904), ('ndcg@5', 0.0822), ('ndcg@10', 0.1284), ('ndcg@15', 0.1432), ('ndcg@20', 0.1535)])
best valid : OrderedDict([('recall@5', 0.1533), ('recall@10', 0.3293), ('recall@15', 0.394), ('recall@20', 0.4478), ('ndcg@5', 0.0893), ('ndcg@10', 0.1436), ('ndcg@15', 0.1608), ('ndcg@20', 0.1735)])
test result: OrderedDict([('recall@5', 0.1415), ('recall@10', 0.2937), ('recall@15', 0.3575), ('recall@20', 0.4019), ('ndcg@5', 0.0794), ('ndcg@10', 0.127), ('ndcg@15', 0.1439), ('ndcg@20', 0.1544)])
best valid : OrderedDict([('recall@5', 0.1477), ('recall@10', 0.3304), ('recall@15', 0.3992), ('recall@20', 0.4532), ('ndcg@5', 0.0886), ('ndcg@10', 0.1449), ('ndcg@15', 0.1631), ('ndcg@20', 0.1759)])
test result: OrderedDict([('recall@5', 0.1351), ('recall@10', 0.2957), ('recall@15', 0.3585), ('recall@20', 0.4011), ('ndcg@5', 0.0787), ('ndcg@10', 0.1293), ('ndcg@15', 0.1459), ('ndcg@20', 0.156)])


## NGCF
best valid : OrderedDict([('recall@5', 0.2848), ('recall@10', 0.4093), ('recall@15', 0.4798), ('recall@20', 0.525), ('ndcg@5', 0.1911), ('ndcg@10', 0.2314), ('ndcg@15', 0.2502), ('ndcg@20', 0.2609)])
test result: OrderedDict([('recall@5', 0.2403), ('recall@10', 0.3427), ('recall@15', 0.3995), ('recall@20', 0.4434), ('ndcg@5', 0.1631), ('ndcg@10', 0.1966), ('ndcg@15', 0.2117), ('ndcg@20', 0.2221)])


## LightGCN
best valid : OrderedDict([('recall@5', 0.318), ('recall@10', 0.4394), ('recall@15', 0.4954), ('recall@20', 0.5336), ('ndcg@5', 0.2211), ('ndcg@10', 0.26), ('ndcg@15', 0.2749), ('ndcg@20', 0.2839)])
test result: OrderedDict([('recall@5', 0.2613), ('recall@10', 0.3582), ('recall@15', 0.4058), ('recall@20', 0.442), ('ndcg@5', 0.179), ('ndcg@10', 0.2108), ('ndcg@15', 0.2235), ('ndcg@20', 0.2321)])

## SGL
best valid : OrderedDict([('recall@5', 0.3343), ('recall@10', 0.4536), ('recall@15', 0.5107), ('recall@20', 0.5498), ('ndcg@5', 0.2249), ('ndcg@10', 0.2639), ('ndcg@15', 0.2791), ('ndcg@20', 0.2884)])
test result: OrderedDict([('recall@5', 0.2877), ('recall@10', 0.3862), ('recall@15', 0.4379), ('recall@20', 0.476), ('ndcg@5', 0.1925), ('ndcg@10', 0.2244), ('ndcg@15', 0.2381), ('ndcg@20', 0.2472)])

best valid : OrderedDict([('recall@5', 0.3618), ('recall@10', 0.4541), ('recall@15', 0.506), ('recall@20', 0.5447), ('ndcg@5', 0.2438), ('ndcg@10', 0.2739), ('ndcg@15', 0.2877), ('ndcg@20', 0.2969)])
test result: OrderedDict([('recall@5', 0.3031), ('recall@10', 0.3826), ('recall@15', 0.4322), ('recall@20', 0.4693), ('ndcg@5', 0.214), ('ndcg@10', 0.2398), ('ndcg@15', 0.2529), ('ndcg@20', 0.2618)])

