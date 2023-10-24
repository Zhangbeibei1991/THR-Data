'''
收集合格的零样本few-shot和zero-shot数据
'''
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
random.seed(42)
np.random.seed(42)
data = pd.read_csv("cache/graph/round_one.csv", sep="\t").values.tolist()

stata_data = {}

for (id, text, event, relation, mix_map, meta) in tqdm(data, desc="sata"):
    text = text.strip()
    if not pd.isna(relation):
        triples = relation.split("\n")
        for triple in triples:
            head, end, rel = triple.split("\t")
            if rel not in stata_data:
                stata_data[rel] = 0
            stata_data[rel] += 1

updated_stata = []
rel2map = {}
stata_data = sorted(stata_data.items(), key=lambda x: x[1], reverse=True)
print("关系映射: ")
for key, count in stata_data:
    print(key, count)
    if count > 10:
        updated_stata.append(key)
        rel2map[key] = len(rel2map)

ann_data = pd.read_csv("ann/store_data.csv", sep="\t").values.tolist()
map2rel = set()
for (id, text, e1_mention, e2_mention, relation) in tqdm(ann_data, desc="collect"):
    map2rel.add((text.strip(), e1_mention, e2_mention, relation))

already_set = set()

filter_data = {"id": [], "sentence": [], "e1_mention": [], "e2_mention": [], "relation": []}
for (id, text, event, relation, mix_map, meta) in tqdm(data, desc="sata"):
    text = text.strip()
    if not pd.isna(relation):
        triples = relation.split("\n")
        for triple in triples:
            head, end, rel = triple.split("\t")
            head_ = head
            end_ = end
            if rel in updated_stata and (text.strip(), head, end, rel) not in map2rel:
                head = head.split("]")[0].strip() + "]"
                end = end.split("]")[0].strip() + "]"
                head_word = head.split("|")[0].strip("[").strip()
                end_word = end.split("|")[0].strip("[").strip()
                check_items = (text, head, end)
                if head_word == end_word:
                    continue
                if check_items not in already_set:
                    already_set.add(check_items)
                    filter_data["id"].append(id)
                    filter_data["sentence"].append(text)
                    # filter_data["e1_mention"].append(head)
                    filter_data["e1_mention"].append(head_)
                    # filter_data["e2_mention"].append(end)
                    filter_data["e2_mention"].append(end_)
                    filter_data["relation"].append(rel)

filter_data_ = pd.DataFrame(filter_data)
filter_data_.to_csv("./cache/FsPreTrainBioERE.csv", sep="\t", index=0)
num_examples = len(filter_data["relation"])
print(f"总的三元组样本数: {num_examples}")
with open("./cache/stata/rel2map.json", mode="w", encoding="utf-8") as f:
    json.dump(rel2map, f, ensure_ascii=False, indent=4)

stata = sorted(dict(Counter(filter_data["relation"])).items(), key=lambda x:x[1], reverse=True)

for key, value in stata:
    print(key, value)

'''
总的三元组样本数: 169467
produces 48081
consists_of 38946
co-occurs_with 38605
occurs_in 12625
associated_with 10291
result_of 6823
prevents 4961
affects 3793
location_of 3112
part_of 1138
interacts_with 506
causes 366
disrupts 220
'''


# rel2instances = {}
#
# for (idx, sentence, e1_mention, e2_mention, relation) in filter_data_.values.tolist():
#     if relation not in rel2instances:
#         rel2instances[relation] = []
#     rel2instances[relation].append((idx, sentence, e1_mention, e2_mention, relation))
#
# valid_ratio = 0.1
# train_Fs = {"id": [], "sentence": [], "e1_mention": [], "e2_mention": [], "relation": []}
# valid_Fs = {"id": [], "sentence": [], "e1_mention": [], "e2_mention": [], "relation": []}

# select_relations = random.sample(rel2map.keys(), 2)
# select_relations = ["disrupts", "affects", "interacts_with", "causes"]
#
# for relation, bag in rel2instances.items():
#     for sample in bag:
#         idx, sentence, e1_mention, e2_mention, relation = sample
#         if relation not in select_relations:
#             train_Fs["id"].append(idx)
#             train_Fs["sentence"].append(sentence)
#             train_Fs["e1_mention"].append(e1_mention)
#             train_Fs["e2_mention"].append(e2_mention)
#             train_Fs["relation"].append(relation)
#         else:
#             valid_Fs["id"].append(idx)
#             valid_Fs["sentence"].append(sentence)
#             valid_Fs["e1_mention"].append(e1_mention)
#             valid_Fs["e2_mention"].append(e2_mention)
#             valid_Fs["relation"].append(relation)
#
# train_Fs_ = pd.DataFrame(train_Fs)
# train_Fs_.to_csv(f"./cache/ZsTrainBioERE.csv", sep="\t", index=0)
# train_Fs_num = len(train_Fs["relation"])
# train_rel_num = len(set(train_Fs["relation"]))
# print(f"train num: {train_Fs_num}")
# print(f"train relation num: {train_rel_num}")
#
# valid_Fs_ = pd.DataFrame(valid_Fs)
# valid_Fs_.to_csv(f"./cache/ZsTestBioERE-1.csv", sep="\t", index=0)
# valid_Fs_num = len(valid_Fs["relation"])
# valid_rel_num = len(set(valid_Fs["relation"]))
# print(f"valid num: {valid_Fs_num}")
# print(f"train relation num: {valid_rel_num}")
# print(set(valid_Fs["relation"]))
