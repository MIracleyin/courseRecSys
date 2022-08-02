# final result

## Pop
best valid : OrderedDict([('recall@5', 0.0903), ('recall@10', 0.1547), ('recall@15', 0.21), ('recall@20', 0.2742), ('ndcg@5', 0.0592), ('ndcg@10', 0.0802), ('ndcg@15', 0.0948), ('ndcg@20', 0.1099)])
test result: OrderedDict([('recall@5', 0.1021), ('recall@10', 0.1699), ('recall@15', 0.2245), ('recall@20', 0.2862), ('ndcg@5', 0.0677), ('ndcg@10', 0.0898), ('ndcg@15', 0.1042), ('ndcg@20', 0.1187)])

## BPR
best valid : OrderedDict([('recall@5', 0.3061), ('recall@10', 0.397), ('recall@15', 0.458), ('recall@20', 0.4986), ('ndcg@5', 0.2263), ('ndcg@10', 0.2556), ('ndcg@15', 0.2719), ('ndcg@20', 0.2815)])
test result: OrderedDict([('recall@5', 0.3134), ('recall@10', 0.4015), ('recall@15', 0.4602), ('recall@20', 0.502), ('ndcg@5', 0.2516), ('ndcg@10', 0.2799), ('ndcg@15', 0.2956), ('ndcg@20', 0.3055)])

## PinSage
best valid : OrderedDict([('recall@5', 0.101), ('recall@10', 0.1865), ('recall@15', 0.2245), ('recall@20', 0.2838), ('ndcg@5', 0.067), ('ndcg@10', 0.094), ('ndcg@15', 0.1041), ('ndcg@20', 0.1182)])
test result: OrderedDict([('recall@5', 0.1058), ('recall@10', 0.1966), ('recall@15', 0.2269), ('recall@20', 0.2831), ('ndcg@5', 0.0713), ('ndcg@10', 0.0999), ('ndcg@15', 0.1078), ('ndcg@20', 0.1213)])

## TPGNN
### using cnn
best valid : OrderedDict([('recall@5', 0.1921), ('recall@10', 0.3084), ('recall@15', 0.3788), ('recall@20', 0.4278), ('ndcg@5', 0.1136), ('ndcg@10', 0.1511), ('ndcg@15', 0.1698), ('ndcg@20', 0.1815)])
test result: OrderedDict([('recall@5', 0.192), ('recall@10', 0.3086), ('recall@15', 0.3775), ('recall@20', 0.4259), ('ndcg@5', 0.1148), ('ndcg@10', 0.1524), ('ndcg@15', 0.1707), ('ndcg@20', 0.1822)])
### without cnn
best valid : OrderedDict([('recall@5', 0.2422), ('recall@10', 0.3585), ('recall@15', 0.4313), ('recall@20', 0.4785), ('ndcg@5', 0.153), ('ndcg@10', 0.1904), ('ndcg@15', 0.2098), ('ndcg@20', 0.221)])
test result: OrderedDict([('recall@5', 0.242), ('recall@10', 0.3569), ('recall@15', 0.4298), ('recall@20', 0.4769), ('ndcg@5', 0.1528), ('ndcg@10', 0.1897), ('ndcg@15', 0.2092), ('ndcg@20', 0.2203)])
### without relu dropout norm
best valid : OrderedDict([('recall@5', 0.2251), ('recall@10', 0.3517), ('recall@15', 0.416), ('recall@20', 0.465), ('ndcg@5', 0.1297), ('ndcg@10', 0.1705), ('ndcg@15', 0.1876), ('ndcg@20', 0.1993)])
test result: OrderedDict([('recall@5', 0.2251), ('recall@10', 0.35), ('recall@15', 0.4136), ('recall@20', 0.4628), ('ndcg@5', 0.1299), ('ndcg@10', 0.1702), ('ndcg@15', 0.1872), ('ndcg@20', 0.1988)])
### without relu dropout
best valid : OrderedDict([('recall@5', 0.2251), ('recall@10', 0.3517), ('recall@15', 0.416), ('recall@20', 0.465), ('ndcg@5', 0.1297), ('ndcg@10', 0.1705), ('ndcg@15', 0.1876), ('ndcg@20', 0.1993)])
test result: OrderedDict([('recall@5', 0.2251), ('recall@10', 0.35), ('recall@15', 0.4136), ('recall@20', 0.4628), ('ndcg@5', 0.1299), ('ndcg@10', 0.1702), ('ndcg@15', 0.1872), ('ndcg@20', 0.1988)])
### without relu norm
best valid : OrderedDict([('recall@5', 0.2251), ('recall@10', 0.3518), ('recall@15', 0.416), ('recall@20', 0.465), ('ndcg@5', 0.1297), ('ndcg@10', 0.1705), ('ndcg@15', 0.1876), ('ndcg@20', 0.1993)])
test result: OrderedDict([('recall@5', 0.2251), ('recall@10', 0.35), ('recall@15', 0.4136), ('recall@20', 0.4628), ('ndcg@5', 0.1299), ('ndcg@10', 0.1702), ('ndcg@15', 0.1871), ('ndcg@20', 0.1988)])
### without relu norm
best valid : OrderedDict([('recall@5', 0.2251), ('recall@10', 0.3518), ('recall@15', 0.416), ('recall@20', 0.465), ('ndcg@5', 0.1297), ('ndcg@10', 0.1705), ('ndcg@15', 0.1876), ('ndcg@20', 0.1993)])
test result: OrderedDict([('recall@5', 0.2251), ('recall@10', 0.35), ('recall@15', 0.4136), ('recall@20', 0.4628), ('ndcg@5', 0.1299), ('ndcg@10', 0.1702), ('ndcg@15', 0.1871), ('ndcg@20', 0.1988)])

## LightGCN
best valid : OrderedDict([('recall@5', 0.3623), ('recall@10', 0.4658), ('recall@15', 0.525), ('recall@20', 0.5655), ('ndcg@5', 0.2659), ('ndcg@10', 0.2993), ('ndcg@15', 0.3151), ('ndcg@20', 0.3248)])
test result: OrderedDict([('recall@5', 0.3704), ('recall@10', 0.4712), ('recall@15', 0.5288), ('recall@20', 0.5674), ('ndcg@5', 0.295), ('ndcg@10', 0.3276), ('ndcg@15', 0.343), ('ndcg@20', 0.3522)])

## SGL
best valid : OrderedDict([('recall@5', 0.3687), ('recall@10', 0.469), ('recall@15', 0.5262), ('recall@20', 0.5685), ('ndcg@5', 0.2727), ('ndcg@10', 0.3053), ('ndcg@15', 0.3205), ('ndcg@20', 0.3305)])
test result: OrderedDict([('recall@5', 0.3768), ('recall@10', 0.474), ('recall@15', 0.5294), ('recall@20', 0.5723), ('ndcg@5', 0.3022), ('ndcg@10', 0.3336), ('ndcg@15', 0.3484), ('ndcg@20', 0.3586)])
