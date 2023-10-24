import pandas as pd
from tqdm import tqdm
import pickle
import os

os.makedirs("./cache/graph", exist_ok=True)

cui2cui_map = pd.read_csv("./cache/base/CUI2CUI.csv")

CUI2CUI_map1 = cui2cui_map.groupby(['CUI1', 'CUI2']).REL.apply(list).to_dict()
CUI2CUI_map = {}
for (head, tail), relations in CUI2CUI_map1.items():
    CUI2CUI_map[(head, tail)] = "|".join(sorted(list(set(relations))))

data = pd.read_csv("./cache/event.csv", sep="\t").values.tolist()
# 后续的改进: 其实抽的实体是有->的符号的, 后续要改成 @ 这个符号避免重叠 | 而且有的实体还没有 \t 需要后续进一步检查
'''建立全局的实体概念关系映射'''
# all_ents = set()
# for i, (doc_id, text, events, mix_map, meta) in tqdm(enumerate(data), desc="收集全局实体"):
#     meta_list = meta.split("\n")
#     new_meta_list = []
#     for meta_element in meta_list:
#         if "\t" not in meta_element:
#             continue
#         if len(meta_element.split("->")[1]) == 0:
#             all_ents.add(meta_element.split("->")[-1].strip())
#         else:
#             all_ents.add(meta_element.split("->")[1].strip())
#
# all_ents = list(all_ents)
# doc_graph_simple, doc_graph_complex = {}, {}
# for i, meta_i in tqdm(enumerate(all_ents), desc="建立全局实体关系图"):
#     try:
#         meta_word_i, meta_type_i, meta_CUIs_i, meta_TUIs_i = meta_i.split("\t")
#     except:
#         continue
#     for j, meta_j in enumerate(all_ents):
#         try:
#             meta_word_j, meta_type_j, meta_CUIs_j, meta_TUIs_j = meta_j.split("\t")
#         except:
#             continue
#         if i < j and meta_word_j != meta_word_j and meta_word_i not in meta_word_j and meta_word_j not in meta_word_i:
#             meta_CUIs_i_ = meta_CUIs_i.split("@")
#             meta_CUIs_j_ = meta_CUIs_j.split("@")
#             word_pair = (meta_word_i, meta_word_j)
#             for cui_i in meta_CUIs_i_:
#                 for cui_j in meta_CUIs_j_:
#                     if (cui_i, cui_j) in CUI2CUI_map:
#                         if word_pair not in doc_graph_simple:
#                             doc_graph_simple[word_pair] = set()
#                             doc_graph_complex[word_pair] = set()
#                         rels = CUI2CUI_map[(cui_i, cui_j)]
#                         doc_graph_complex[word_pair].add(f"{meta_word_i}:{cui_i}-{cui_j}:{meta_word_j}->" + rels)
#                         doc_graph_simple[word_pair].update(set(rels.split("|")))
#
# with open("./cache/all_graph.pt", mode="wb") as f:
#     pickle.dump([doc_graph_simple,doc_graph_complex], f)
# print(f"全局图中远程监督实体三元组数目: {len(doc_graph_simple)}")

ent_graph = {}
doc_num = 0
triple_count = set()
for i, (doc_id, text, events, mix_map, meta) in tqdm(enumerate(data), desc="建立文档级实体关系图"):
    if pd.isna(meta):
        meta_list = []
    else:
        meta_list = meta.split("\n")
    doc_graph_simple, doc_graph_complex = {}, {}
    for meta_i, each_meta_i in enumerate(meta_list):
        meta_word_i, meta_type_i, meta_CUIs_i, meta_TUIs_i = each_meta_i.split(":")[-1].split("\t")
        meta_flag_i = each_meta_i.split("##")[0].split(":")[0]
        for meta_j, each_meta_j in enumerate(meta_list):
            meta_word_j, meta_type_j, meta_CUIs_j, meta_TUIs_j = each_meta_j.split(":")[-1].split("\t")
            meta_flag_j = each_meta_j.split("##")[0].split(":")[0]
            if meta_i < meta_j and meta_word_i != meta_word_j and meta_word_i not in meta_word_j and meta_word_j not in meta_word_i and meta_word_i.replace(
                    " ", "") != meta_word_j.replace(" ", ""):
                meta_CUIs_i_ = meta_CUIs_i.split("@")
                meta_CUIs_j_ = meta_CUIs_j.split("@")
                word_pair = (meta_flag_i + ":" + meta_word_i, meta_flag_i + ":" + meta_word_j)

                for cui_i in meta_CUIs_i_:
                    for cui_j in meta_CUIs_j_:
                        if (cui_i, cui_j) in CUI2CUI_map:
                            if word_pair not in doc_graph_simple:
                                doc_graph_simple[word_pair] = set()
                                doc_graph_complex[word_pair] = set()
                            rels = CUI2CUI_map[(cui_i, cui_j)]
                            doc_graph_complex[word_pair].add(f"{meta_word_i}:{cui_i}-{cui_j}:{meta_word_j}->" + rels)
                            doc_graph_simple[word_pair].update(set(rels.split("|")))
                            triple_count.add(f"{meta_word_i}:{cui_i}-{cui_j}:{meta_word_j}->" + rels)
    ent_graph[doc_id] = [doc_graph_simple, doc_graph_complex]
    doc_num += len(doc_graph_simple)
print(f"triple num: {len(triple_count)}")
with open("./cache/graph/doc_graph.pt", mode="wb") as f:
    pickle.dump(ent_graph, f)

print(f"文档图中远程监督实体三元组数目: {doc_num}")

'''
triple num: 1387950
文档图中远程监督实体三元组数目: 5557729
'''