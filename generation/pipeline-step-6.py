import pandas as pd
from tqdm import tqdm
from collections import Counter
import json
import os
import pickle
from collections import defaultdict

'''
1. 收集文本中所有映射的CUI和TUI
2. 构建UMLS知识库中关于CUI和TUI的映射文件(.csv)
'''

os.makedirs("./cache/base", exist_ok=True)

data_dir = "./cache/event.csv"

data = pd.read_csv(data_dir, sep="\t").values.tolist()

'''
收集文本中所有CUI和STY集合
'''

all_CUIs, all_TUIs = set(), set()
eae = json.load(open("cache/stata/map.json", encoding="utf-8", mode="r"))
evt_list, arg_list, ent_list = eae["event"], eae["argument"], eae["entity"]

count = 0
for (doc_id, text, evts, mix_map, meta) in tqdm(data, desc="collect Entity and Map"):
    if pd.isna(meta):
        metas = []
    else:
        metas = meta.split("\n")
    for unit in metas:
        _, _, CUIs, TUIs = unit.split("\t")
        CUIs = set(CUIs.split("|"))
        TUIs = set(TUIs.split("|"))
        for CUI in CUIs:
            all_CUIs.update(set(CUI.split("@")))
        for TUI in TUIs:
            all_TUIs.update(set(TUI.split("@")))

with open("./cache/base/node.json", encoding="utf-8", mode="w") as f:
    json.dump({"CUI": list(all_CUIs), "TUI": list(all_TUIs)}, f, indent=4,
              ensure_ascii=False)

print(f"文本中 CUI 和 TUI 的数目: {len(all_CUIs)} and {len(all_TUIs)}")
'''文本中 CUI 和 TUI 的数目: 110201 and 128'''

if not os.path.exists("./cache/base/CUI2CUI.csv"):
    Bulk_num = 10000000
    with open("../UMLS2020AA/MRREL.RRF", encoding='utf-8', mode='r') as f:
        for i, _ in enumerate(f.readlines(), 1):
            Total_num = i

    print('在 MRREL.RRF 中数据的条数: ', Total_num)
    part_id = 0
    lines = []
    with open("../UMLS2020AA/MRREL.RRF", encoding='utf-8', mode='r') as f:
        for i, line in tqdm(enumerate(f.readlines(), 1), desc="读取 MRREL.RRF"):
            lines.append([line.split('|')[0], line.split('|')[3], line.split('|')[4], line.split('|')[7]])
            if len(lines) == Bulk_num or i == Total_num:
                CUI1_REL_CUI2_RELA = pd.DataFrame(lines, columns=['CUI1', 'REL', 'CUI2', 'RELA'])
                CUI1_REL_CUI2_RELA.to_csv(os.path.join("./cache/base", 'CUI1_REL_CUI2_RELA={}.csv'.format(part_id)),
                                          index=None)
                print('保存 .csv 临时文件: ', 'CUI1_REL_CUI2_RELA={}.csv'.format(part_id))
                part_id += 1
                lines = []

    '''首先确定全局的cui_neighbor_counter的统计结果'''
    cache_list = [item for item in os.listdir("./cache/base") if "CUI1_REL_CUI2_RELA=" in item]
    CUIs = all_CUIs
    all_eCUIs = []
    threshold = 35  # 原代码是35
    for i, cache_name in enumerate(cache_list):
        cache_path = os.path.join("./cache/base/", cache_name)
        split_csv = pd.read_csv(cache_path)
        filter_split_csv = split_csv.loc[(split_csv.CUI1.isin(CUIs) | split_csv.CUI2.isin(CUIs))]
        all_eCUIs.extend(filter_split_csv.CUI1.tolist() + filter_split_csv.CUI2.tolist())

    cui_neighbor_counter = Counter(all_CUIs)
    filtered_cuis = [cui for cui, count in cui_neighbor_counter.items() if count > threshold]  # 去除低频的cui
    selected_cuis = CUIs.union(filtered_cuis)  # filter_split_csv 因为是或的关系,所以可能会引入别的cui

    with open('./cache/base/neighbors.pt', mode='wb') as f:
        pickle.dump([cui_neighbor_counter, selected_cuis], f)
    print("新发现的 CUI 邻居数: ", len(cui_neighbor_counter))
    print("最终确定的 CUI 数目: ", len(selected_cuis))
    print('保存扩展 CUI 为 neighbors.pt !')

    merge_csv = None
    for i, cache_name in enumerate(cache_list):
        cache_path = os.path.join("./cache/base", cache_name)
        split_csv = pd.read_csv(cache_path)
        filtered_split_csv = split_csv.loc[(split_csv.CUI1.isin(selected_cuis)) & (split_csv.CUI2.isin(selected_cuis))]
        if i == 0:
            merge_csv = filtered_split_csv
        else:
            merge_csv = pd.concat([merge_csv, filtered_split_csv], ignore_index=True)
        os.remove(cache_path)
    merge_csv = merge_csv.drop_duplicates()
    merge_csv.to_csv("./cache/base/CUI2CUI.csv", index=False)

if not os.path.exists("./cache/base/CUI2STY.csv"):
    with open("../UMLS2020AA/MRSTY.RRF", encoding='utf-8', mode='r') as f:
        lines = [line.split('|')[:-1] for line in f.readlines()]
    CUI_STY = pd.DataFrame(lines, columns=['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF'])
    CUI_STY.to_csv('./cache/base/CUI2STY.csv', index=None)
    print('保存 CUI2STY.csv !')

if not os.path.exists("./cache/base/STY2STY.csv"):
    lines = []
    with open("../UMLS2020AA/SRSTR", encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            lines.append(line.split('|')[:-1])
    STYRL1_RL_STYRL2_LS = pd.DataFrame(lines, columns=['STY1', 'RL', 'STY2', 'LS'])
    STYRL1_RL_STYRL2_LS.to_csv('./cache/base/STY2STY.csv', index=None)
    print('保存 STY2STY.csv !')

if not os.path.exists("./cache/base/CUI2STR.csv"):
    lines = []
    with open("../UMLS2020AA/MRCONSO.RRF", encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            lines.append([line.split('|')[0], line.split('|')[6], line.split('|')[14]])
    CUI_STR = pd.DataFrame(lines, columns=['CUI', 'ISPREF', 'STR'])
    CUI_STR.to_csv('./cache/base/CUI2STR.csv', index=None)
    print('保存 CUI2STR.csv !')

CUI2CUI = pd.read_csv(os.path.join('./cache/base/CUI2CUI.csv'))
STY2STY = pd.read_csv(os.path.join('./cache/base/STY2STY.csv'))
CUI2STY = pd.read_csv(os.path.join('./cache/base/CUI2STY.csv'))
_, selected_cuis = pickle.load(open('./cache/base/neighbors.pt', mode="rb"))

# construct REL to idx mapping
REL2idx = {REL: idx for idx, REL in enumerate(sorted(set(CUI2CUI.REL)))}
idx2REL = {idx: REL for idx, REL in enumerate(sorted(set(CUI2CUI.REL)))}

# construct RL to idx mapping
RL2idx = {RL: idx for idx, RL in enumerate(sorted(STY2STY.RL.unique()))}  # unique 类似于字典
idx2RL = {idx: RL for idx, RL in enumerate(sorted(STY2STY.RL.unique()))}

# construct CUI to idx mapping 这个地方的设定是非常奇怪的
CUI2idx = {cui: idx for idx, cui in enumerate(sorted(selected_cuis.union(set(CUI2STY.STY))))}
idx2CUI = {idx: cui for idx, cui in enumerate(sorted(selected_cuis.union(set(CUI2STY.STY))))}

# construct STY to idx mapping
STY2idx = {sty: idx for idx, sty in enumerate(sorted(set(CUI2STY.STY)))}
idx2STY = {idx: sty for idx, sty in enumerate(sorted(set(CUI2STY.STY)))}

# construct REL + RL to idx mapping
RELRL2idx = {relation: idx for idx, relation in enumerate(list(REL2idx.keys()) + list(RL2idx.keys()))}
idx2RELRL = {idx: relation for idx, relation in enumerate(list(REL2idx.keys()) + list(RL2idx.keys()))}

CUI2CUI_map = CUI2CUI.groupby(['CUI1', 'CUI2']).REL.apply(list).to_dict()
CUI_adj_list = defaultdict(set)
for CUI1, CUI2 in CUI2CUI_map.keys():
    CUI_adj_list[CUI1].add(CUI2)
    CUI_adj_list[CUI2].add(CUI1)

all_STYs = sorted(set(CUI2STY.STY))
fine_grained_map = {'REL2idx': REL2idx, 'idx2REL': idx2REL, 'RL2idx': RL2idx, 'idx2RL': idx2RL,
                    'CUI2idx': CUI2idx, 'idx2CUI': idx2CUI, 'RELRL2idx': RELRL2idx, 'idx2RELRL': idx2RELRL,
                    'all_STYs': all_STYs, 'CUI_adj_list': CUI_adj_list}

with open(os.path.join('./cache/base/kg_map.pt'), mode='wb') as f:
    pickle.dump(fine_grained_map, f)
print('保存细粒度映射字典 kg_map.pt ...')
