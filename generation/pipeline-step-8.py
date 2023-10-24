'''获得(e1_type, relation, e2_type)与关系映射的三元组'''
import json
import pickle
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np
import random

random.seed(42)

target_relations = ["result_of", 'co-occurs_with', 'occurs_in', 'consists_of', 'affects', 'associated_with',
                    'location_of', 'exhibits', 'causes', 'produces', 'process_of', 'disrupts', 'connected_to',
                    'adjacent_to', 'branch_of', 'tributary_of', 'surrounds', 'part_of', 'interconnects',
                    'derivative_of', 'interacts_with', 'precedes', 'manages', 'prevents']

os.makedirs("./cache/graph", exist_ok=True)

if not os.path.exists("./cache/graph/cands.json"):
    tui2sty_map = pd.read_csv("./cache/base/CUI2STY.csv")
    cui2cui_map = pd.read_csv("./cache/base/CUI2CUI.csv")
    sty2sty_map = pd.read_csv("./cache/base/STY2STY.csv")
    data = pd.read_csv("./cache/event.csv", sep="\t").values.tolist()
    kg_map = pickle.load(open("./cache/graph/doc_graph.pt", mode="rb"))
    TUI2STY_map = tui2sty_map.groupby('TUI').STY.apply(set).to_dict()
    CUI2CUI_map = cui2cui_map.groupby(['CUI1', 'CUI2']).REL.apply(set).to_dict()
    STY2STY_map = sty2sty_map.groupby(['STY1', 'STY2']).RL.apply(set).to_dict()

    collect_r_triple = {}
    collect_wr_triple = {}
    argument_r_path = {}
    argument_wr_path = {}
    for i, (doc_id, text, events, mix_map, meta) in tqdm(enumerate(data), desc="收集三元组"):
        if pd.isna(mix_map):
            mix_map_list = []
        else:
            mix_map_list = mix_map.split("\n")

        if pd.isna(events):
            events = []
        else:
            event_list = events.split("\n")

        event_maps = list(set([item for item in mix_map_list if item.endswith("Event")]))

        for ii, item_i in enumerate(event_maps):
            for jj, item_j in enumerate(event_maps):
                if ii < jj and item_i != item_j:
                    match_evt_i, _, _, CUIs_i, TUIs_i, _, _ = item_i.split("\t")
                    match_evt_j, _, _, CUIs_j, TUIs_j, _, _ = item_j.split("\t")

                    if CUIs_i == "NA" or CUIs_j == "NA":
                        continue

                    evt_i_w = match_evt_i.split("|")[0].strip("[").strip()
                    evt_i_type = match_evt_i.split("|")[-1].strip(']').strip()
                    evt_j_type = match_evt_j.split("|")[-1].strip(']').strip()
                    evt_j_w = match_evt_j.split("|")[0].strip("[").strip()

                    TUIs_i_ = TUIs_i.split("|")
                    TUIs_j_ = TUIs_j.split("|")

                    rel_pattern = []
                    for ki, TUI_i in enumerate(TUIs_i_):
                        for kj, TUI_j in enumerate(TUIs_j_):
                            if TUI_i not in TUI2STY_map or TUI_j not in TUI2STY_map:
                                continue
                            STY_i = list(TUI2STY_map[TUI_i])[0]
                            STY_j = list(TUI2STY_map[TUI_j])[0]
                            if (STY_i, STY_j) in STY2STY_map:
                                RL = STY2STY_map[(STY_i, STY_j)]
                                rel_pattern.extend([item for item in list(RL) if item in target_relations])
                    if len(rel_pattern) > 0:
                        key1 = (evt_i_type, evt_j_type)
                        if key1 not in collect_r_triple:
                            collect_r_triple[key1] = []
                            argument_r_path[key1] = []
                        collect_r_triple[key1].extend(rel_pattern)

                        key2 = (evt_i_type + ":" + evt_i_w, evt_j_w + ":" + evt_j_type)
                        if key2 not in collect_wr_triple:
                            collect_wr_triple[key2] = []
                            argument_wr_path[key2] = []
                        collect_wr_triple[key2].extend(rel_pattern)

        '''寻找要素语义路径'''
        for e_i, event_i in enumerate(event_list):
            word_span_i, word_type_i = event_i.split("]")[0].strip("[").split("|")
            word_span_i, word_type_i = word_span_i.strip(), word_type_i.strip()
            span_i_set = set()
            for item in mix_map_list:
                match_evt, _, _, _, TUIs, _, _ = item.split("\t")
                if match_evt in event_i:
                    span_i_set.update(set(TUIs.split("|")))
            for e_j, event_j in enumerate(event_list):
                if e_i < e_j and event_i != event_j:
                    word_span_j, word_type_j = event_j.split("]")[0].strip("[").split("|")
                    word_span_j, word_type_j = word_span_j.strip(), word_type_j.strip()
                    span_j_set = set()
                    for item in mix_map_list:
                        match_evt, _, _, _, TUIs, _, _ = item.split("\t")
                        if match_evt in event_j:
                            span_j_set.update(set(TUIs.split("|")))

                    key1 = (word_type_i, word_type_j)
                    key2 = (word_type_i + ":" + word_span_i, word_span_j + ":" + word_type_j)
                    for ii, item_i in enumerate(span_i_set):
                        for jj, item_j in enumerate(span_j_set):
                            if ii < jj:
                                if key1 not in argument_r_path:
                                    argument_r_path[key1] = []
                                if key2 not in argument_wr_path:
                                    argument_wr_path[key2] = []
                                if item_i == "NA" or item_j == "NA":
                                    continue
                                for TUI_a in item_i.split("@"):
                                    for TUI_b in item_j.split("@"):
                                        STY_i = list(TUI2STY_map[TUI_a])[0]
                                        STY_j = list(TUI2STY_map[TUI_b])[0]

                                        if (STY_i, STY_j) in STY2STY_map:
                                            argument_r_path[key1].extend(
                                                [item for item in list(STY2STY_map[(STY_i, STY_j)]) if
                                                 item in target_relations])
                                            argument_wr_path[key2].extend(
                                                [item for item in list(STY2STY_map[(STY_i, STY_j)]) if
                                                 item in target_relations])

    updated_collect_r_triple = {}
    for key, value in collect_r_triple.items():
        if len(value) == 0:
            continue
        updated_collect_r_triple[f"{key[0]}<->{key[1]}"] = [f"{item[0]}@{item[1]}" for item in
                                                            sorted(dict(Counter(value)).items(), key=lambda x: x[1],
                                                                   reverse=True)]

    updated_collect_wr_triple = {}
    for key, value in collect_wr_triple.items():
        if len(value) == 0:
            continue
        updated_collect_wr_triple[f"{key[0]}<->{key[1]}"] = [f"{item[0]}@{item[1]}" for item in
                                                             sorted(dict(Counter(value)).items(), key=lambda x: x[1],
                                                                    reverse=True)]

    updated_argument_r_path = {}
    for key, value in argument_r_path.items():
        if len(value) == 0:
            continue
        updated_argument_r_path[f"{key[0]}<->{key[1]}"] = [f"{item[0]}@{item[1]}" for item in
                                                           sorted(dict(Counter(value)).items(), key=lambda x: x[1],
                                                                  reverse=True)]

    updated_argument_wr_path = {}
    for key, value in argument_wr_path.items():
        if len(value) == 0:
            continue
        updated_argument_wr_path[f"{key[0]}<->{key[1]}"] = [f"{item[0]}@{item[1]}" for item in
                                                            sorted(dict(Counter(value)).items(), key=lambda x: x[1],
                                                                   reverse=True)]

    with open("./cache/graph/cands.json", mode="w", encoding="utf-8") as f:
        json.dump([updated_collect_r_triple, updated_collect_wr_triple,
                   updated_argument_r_path, updated_argument_wr_path], f, indent=4, ensure_ascii=False)

else:
    updated_collect_r_triple, updated_collect_wr_triple, updated_argument_r_path, updated_argument_wr_path = json.load(
        open("./cache/graph/cands.json"))

'''
筛选要素路径样例, 将样本总数少于某个阈值的舍去
'''

etp_set = set()
for value in updated_argument_wr_path.values():
    for item in value:
        etp_set.add(item.split("@")[0])

etp = len(etp_set)
etp_dict = {k: i for i, k in enumerate(etp_set)}
temp_awr_path = {}
for i, (key, value) in enumerate(updated_argument_wr_path.items()):
    instances = [(item.split("@")[0], int(item.split("@")[1])) for item in value]
    int_sum = sum([item[1] for item in instances])
    if int_sum > 0:
        temp_awr_path[key] = value

if not os.path.exists("./cache/graph/etp_count.json"):
    count_etp_j = {}
    for key in tqdm(etp_set, desc="收集要各素路径总径数"):
        for key_s, value_s in temp_awr_path.items():
            for item in value_s:
                if item.split("@")[0] == key:
                    if key not in count_etp_j:
                        count_etp_j[key] = 0
                    count_etp_j[key] += int(item.split("@")[1])
                    break
    with open("./cache/graph/etp_count.json", encoding="utf-8", mode="w") as f:
        json.dump(count_etp_j, f, indent=4, ensure_ascii=False)
else:
    count_etp_j = json.load(open("./cache/graph/etp_count.json", encoding="utf-8", mode="r"))

match_path = {}
for i, (key, value) in tqdm(enumerate(temp_awr_path.items()), desc="路径匹配算法"):
    instances = [(item.split("@")[0], int(item.split("@")[1])) for item in value]
    int_sum = sum([item[1] for item in instances])
    aps_i = []
    for j, int_item in enumerate(instances):
        '''计算要素路径重要性'''
        count_pa_i_etp_j = int_item[1]
        aps_ij = count_pa_i_etp_j / (count_etp_j[int_item[0]] / len(temp_awr_path) + 1e-10)
        aps_i.append(aps_ij)
    '''触发词语义匹配频率'''
    etps_i = len(instances)
    tpmf_i = np.tanh(etps_i / (etp + 1e-10))
    tmr_ij = [item * tpmf_i for item in aps_i]
    results = [f"{aim}@{0.0}" for aim in etp_dict.keys()]
    for j, int_item in enumerate(instances):
        results[etp_dict[int_item[0]]] = f"{int_item[0]}@{aps_i[j]}"
    results = sorted(results, key=lambda x: float(x.split("@")[1]), reverse=True)[:3]
    match_path[key] = '_$_'.join([item.split("@")[0] for item in results if float(item.split("@")[1]) > 0.0])

'''
筛选关系魔板样例, 将样本总数少于某个阈值的舍去
'''

etp_set = set()
for value in updated_collect_wr_triple.values():
    for item in value:
        etp_set.add(item.split("@")[0])

etp = len(etp_set)
etp_dict = {k: i for i, k in enumerate(etp_set)}
temp_awr_path = {}
for i, (key, value) in enumerate(updated_collect_wr_triple.items()):
    instances = [(item.split("@")[0], int(item.split("@")[1])) for item in value]
    int_sum = sum([item[1] for item in instances])
    if int_sum > 0:
        temp_awr_path[key] = value

count_etp_j = {}
for key in etp_set:
    for key_s, value_s in temp_awr_path.items():
        for item in value_s:
            if item.split("@")[0] == key:
                if key not in count_etp_j:
                    count_etp_j[key] = 0
                count_etp_j[key] += int(item.split("@")[1])
                break

trigger_path = {}

for i, (key, value) in tqdm(enumerate(temp_awr_path.items()), desc="路径匹配算法"):
    instances = [(item.split("@")[0], int(item.split("@")[1])) for item in value]
    int_sum = sum([item[1] for item in instances])
    aps_i = []
    for j, int_item in enumerate(instances):
        '''计算触发词对候选频率'''
        count_pa_i_etp_j = int_item[1]
        aps_ij = count_pa_i_etp_j / (count_etp_j[int_item[0]] / len(temp_awr_path) + 1e-10)
        aps_i.append(aps_ij)
    '''触发词语义匹配频率'''
    etps_i = len(instances)
    temp = etps_i / (etp + 1e-10)
    tpmf_i = np.tanh(etps_i / (etp + 1e-10))
    tmr_ij = [item * tpmf_i for item in aps_i]
    results = [f"{aim}@{0.0}" for aim in etp_dict.keys()]
    for j, int_item in enumerate(instances):
        results[etp_dict[int_item[0]]] = f"{int_item[0]}@{aps_i[j]}"
    results = sorted(results, key=lambda x: float(x.split("@")[1]), reverse=True)
    trigger_path[key] = '_$_'.join([item.split("@")[0] for item in results if float(item.split("@")[1]) > 0.0])
    # trigger_path[key] = '_$_'.join([item.split("@")[0] for item in results if float(item.split("@")[1]) > 30.0])

final_match = {}
final_score = {}
for key, v1 in trigger_path.items():
    v1_list = v1.split("_$_")
    v1_set = set(v1_list)
    if key not in match_path:
        continue
    v2_list = match_path[key].split("_$_")
    v2_set = set(v2_list)
    common_set = sorted(list(v2_set & v1_set))
    best_index, best_rel = 100, None
    for key1 in common_set:
        if key1 in v1_list:
            if v1_list.index(key1) < best_index:
                best_index = v1_list.index(key1)
                best_rel = key1
    if len(common_set) > 0 and len(v1_set) == 1:
        final_match[key] = best_rel
    # if len(common_set) == 1 and len(v2_set) == 1 and len(v1_set) == 1:
    #     final_match[key] = v1_list[0]
print(f"关系三元组数目: {len(final_match)}")
event_graph = {}

data = pd.read_csv("./cache/event.csv", sep="\t").values.tolist()
save_data = {"id": [], "text": [], "event": [], "relation": [], "mix_map": [], "meta": []}
for i, (doc_id, text, events, mix_map, meta) in tqdm(enumerate(data), desc="匹配关系"):
    if pd.isna(mix_map):
        mix_map_list = []
    else:
        mix_map_list = mix_map.split("\n")
    if pd.isna(events):
        event_list = []
    else:
        event_list = events.split("\n")
    event_triples = []
    filter_event_list = []
    remove_event = set()
    for event in event_list:
        event_clean = event.replace(" ", "")
        if event_clean not in remove_event:
            remove_event.add(event_clean)
            filter_event_list.append(event)

    event_list = filter_event_list
    for e_i, event_i in enumerate(event_list):
        word_span_i, word_type_i = event_i.split("]")[0].strip("[").split("|")
        word_span_i, word_type_i = word_span_i.strip(), word_type_i.strip()
        for e_j, event_j in enumerate(event_list):
            if e_i < e_j and event_i != event_j:
                word_span_j, word_type_j = event_j.split("]")[0].strip("[").split("|")
                word_span_j, word_type_j = word_span_j.strip(), word_type_j.strip()
                span_j_set = set()

                key = "<->".join([word_type_i + ":" + word_span_i, word_span_j + ":" + word_type_j])
                if key not in final_match:
                    continue
                rel = final_match[key]

                event_triple = f"{event_i}\t{event_j}\t{rel}"
                if event_triple not in event_triples:
                    event_triples.append(event_triple)

                    if event_i not in event_graph:
                        event_graph[event_i] = set()
                    event_graph[event_i].add(event_j)

    save_data["id"].append(doc_id)
    save_data["text"].append(text)
    save_data["event"].append(events)
    save_data["mix_map"].append(mix_map)
    save_data["meta"].append(meta)
    save_data["relation"].append("\n".join(event_triples))

save_data_ = pd.DataFrame(save_data)
print("数据量: ", len(save_data["relation"]))
save_data_.to_csv("./cache/graph/round_one.csv", sep="\t", index=0)

'''
收集要各素路径总径数: 100%|██████████| 24/24 [00:02<00:00, 11.51it/s]
路径匹配算法: 347046it [00:04, 77465.63it/s]
路径匹配算法: 28846it [00:00, 66618.96it/s]
匹配关系: 2947372it [00:26, 109906.84it/s]
'''

