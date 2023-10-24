import copy
import json
import re
from copy import deepcopy
from tqdm import tqdm
import os
import pandas as pd
from utils import get_std_set
from utils import Match_blankets
from utils import filter_property
from utils import getNumofCommonSubstr

std_event_type, std_argument_type, std_entity_type = get_std_set(root_path="./biot2e")

new_std_argument_type = set()
for item in std_argument_type:
    if item[-1].isdigit():
        new_std_argument_type.add(item[:-1])
    else:
        new_std_argument_type.add(item)

std_argument_type = new_std_argument_type

print(f"事件数: {len(std_event_type)}, 要素数: {len(std_argument_type)}, 实体数: {len(std_entity_type)}")

data_names = [item for item in os.listdir("./sentence") if "-Event" in item and "-Entity" not in item]

mix_types = set()
mix_types.update(std_event_type)
mix_types.update(std_entity_type)
doc_data = {}
last_end_char = 0
event_set, entity_set, argument_set = set(), set(), set()
data_dir = "./sentence"
event_count = {k: 0 for k in std_event_type}
entity_count = {k: 0 for k in std_entity_type}
event_map_sata = {}
total_events, total_entities = set(), set()

'''
print(f"no_NA_sent: {no_NA_sent}, NA_sent: {NA_sent}, NA_rate: {round(NA_sent / (NA_sent + no_NA_sent), 4)}")
=> no_NA_sent: 2947374, NA_sent: 2940522, NA_rate: 0.4994|不含事件句子数: 2947374, 含事件句子数: 2940522, 总句子数: 5887896
'''
NA_sent = 0
no_NA_sent = 0
sent_num = 0
total_data = {"id": [], "text": [], "event": [], "mix_map": [], "meta": []}
events_save = set()
for data_name in data_names:
    flag = False
    data_event = pd.read_csv(os.path.join(data_dir, data_name), sep="\t").values.tolist()
    data_entity = pd.read_csv(os.path.join(data_dir, data_name.replace("-Event", "-Entity")), sep="\t").values.tolist()
    for items_event, items_entity in tqdm(zip(data_event, data_entity), desc=data_name):
        try:
            info, text, text_label = items_event
            _, _, _, entity_label, item_label = items_entity
            lower_text = text.lower()
            doc_id = info.split(".s")[0]

            if pd.isna(text_label):
                NA_sent += 1
                continue
            no_NA_sent += 1

            if pd.isna(entity_label):
                ents = []
            else:
                ents = entity_label.split("$$")
            if pd.isna(item_label):
                terms = []
            else:
                terms = item_label.split("$$")

            '''修正实体类型, '''
            meta_map = set()
            add_types = set()
            for ent_i, ent in enumerate(ents):
                sp_type, sp_offset, sp_text, CUIs, TUIs = ent.split("\t")
                if sp_type not in std_entity_type:
                    flag = True
                    select_terms = list()
                    for official_type in std_entity_type:
                        common_str = getNumofCommonSubstr(str1=official_type.lower(), str2=sp_type.lower())
                        p = common_str[1] / len(official_type)
                        r = common_str[1] / len(sp_type)
                        f = 2 * p * r / (p + r + 1e-10)
                        if f > 0.5:
                            select_terms.append([f, official_type])
                            flag = False
                            break
                    if flag:
                        add_types.add(sp_type)
                    else:
                        select_terms_ = sorted(select_terms, key=lambda x: x[0], reverse=True)
                        sp_type = select_terms_[0][-1]
                start_, end_ = sp_offset.split()
                meta_map.add((f"scispacy:##{sp_text}##",
                              sp_type, CUIs, TUIs))
            std_entity_type.update(add_types)

            for ent_i, ent in enumerate(terms):
                start_, end_, sp_text, sp_type, CUIs, TUIs = ent.split("\t")
                meta_map.add((f"umls:##{sp_text}##",
                              sp_type.capitalize(), CUIs, TUIs))

            '''处理事件'''
            text_label = text_label.replace(" ]", "]").replace("[ ", "[")
            text_label = filter_property(text_label.replace(", |", " |").replace(",]", "]").replace(", ]", "]"))
            events = text_label.replace("] [", "]   [").replace("\n", "").replace("[ ", "[").replace(" ]", "]").split(
                "   ")
            split_events = set()
            event_elements = set()
            for i, event in enumerate(events):
                '''1. 删除事件末尾生成不完全的部分'''
                if not event.endswith("]"):
                    continue
                # '''2. 删除要素的数字标签'''
                new_event = event[0]
                for i in range(1, len(event)):
                    if event[i:].startswith(" = ") and event[i - 1].isdigit():
                        new_event = new_event[:-1]
                    else:
                        new_event += event[i]

                '''3. 对事件进行分块, 切分成最小单元'''
                char_starts = [item.start() for item in re.finditer("\[", event)]
                event_len = len(event)

                overlap_pos = set()
                for char_start in char_starts:
                    if char_start - 1 > 0 and event[char_start - 1] not in ["[", "]"]:
                        char_end = event[char_start:].index("]") + char_start
                        if char_end + 1 < event_len and event[char_end + 1] not in ["[", "]"]:
                            overlap_pos.update(set(list(range(char_start, char_end + 1))))
                            event_part = event[char_start:char_end + 1]

                '''位置检查时加入上实体提及中]的位置, 防止遗漏'''
                for s, each_char in enumerate(event):
                    if 1 < s < event_len - 1 and each_char == "]" and \
                            event[s - 1] not in ["[", "]"] and event[s + 1] not in ["[", "]"]:
                        overlap_pos.add(s)
                event_element = set()
                for char_start in char_starts:
                    if char_start + 1 < event_len and event[char_start + 1] in ["["]:
                        continue
                    char_end = char_start + 1
                    while char_end < event_len:
                        if event[char_end] in ["]"] and char_start not in overlap_pos and char_end not in overlap_pos:
                            char_end += 1
                            event_element.add(event[char_start:char_end])
                            break
                        else:
                            char_end += 1

                '''4. 对嵌套事件进行分割'''
                split_events.add(event)
                event_elements.update(event_element)
                event_parts = []
                if "]]" in event or "[[" in event:
                    left_list, right_list, = [], []
                    if event.endswith("]]"):
                        right_list.append(-1)
                    for k, char_head in enumerate(event):
                        if event[k:].startswith("[["):
                            left_list.append(k)
                        if event[:k].endswith("]]"):
                            right_list.append(k)
                    enumerate_spans = []
                    for s in left_list:
                        for k in right_list:
                            if k != -1 and s < k:
                                span = event[s:k]
                                if len(span.split("][[")) > 1 and span.split("][[")[0].endswith("]"):
                                    continue
                                try:
                                    enumerate_spans.append([Match_blankets(span), s, k, span])
                                except:
                                    # print("异常1")
                                    continue
                            if k == -1:
                                span = event[s:]
                                if len(span.split("][[")) > 1 and span.split("][[")[0].endswith("]"):
                                    continue
                                try:
                                    enumerate_spans.append([Match_blankets(span), s, k, span])
                                except:
                                    # print("异常2")
                                    continue
                    # 检查重叠, 即: A + B = AB, A + B + C = ABC
                    reset, construct = [], []
                    for flag, s, k, mention in enumerate_spans:
                        if flag == 0:
                            reset.append(mention)
                    for A in reset:
                        for B in reset:
                            construct.append(f"{A}{B}")
                            for C in reset:
                                construct.append(f"{A}{B}{C}")
                    for m, item in enumerate(enumerate_spans):
                        if item[-1] in construct:
                            enumerate_spans[m][0] = -1
                    # print(f"**{event}")
                    for flag, s, k, mention in enumerate_spans:
                        if flag == 0:
                            # print(f">>>>{flag}-{s}-{k}: {mention}")
                            # 去除事件触发词的要素标记
                            left_signal = mention.index("]")
                            right_signal = len(mention)
                            for i in reversed(range(len(mention[:left_signal]))):
                                if mention[i] == "|":
                                    right_signal = i
                                    break
                            part = mention[right_signal:left_signal]
                            updated_mention = ""
                            for i, char_each in enumerate(mention):
                                if right_signal <= i < left_signal:
                                    pass
                                else:
                                    updated_mention += char_each
                            mention = updated_mention.replace(" ]", "]")[1:-1]
                            if mention not in split_events:
                                split_events.add(mention)
            '''
            #5. 根据事件内抽取的词是否再文本中，不存在的事件舍去
            #6. 根据事件类型是否在事件中, 如果能修复就修复, 修复不了就舍去
            #注意, 这里不在纠结事件和实体的位置信息, 真正的为知识库抽事件和实体和这个无关, 正好与生成模型结合起来
            '''
            element2map = {}
            for k, element in enumerate(event_elements):
                element_parts = element.split("|")
                if element_parts[0] == "[":
                    split_events = set([item.replace(element, "") for item in split_events])
                    continue
                if len(element_parts) == 1:
                    split_events = set([item.replace(element, "") for item in split_events])
                    continue
                assert len(element_parts) > 1
                if element not in element2map:
                    element2map[element] = True
                if len(element_parts) == 2:
                    if element_parts[0].strip().endswith("["):
                        split_events = set([item.replace(element, "") for item in split_events])
                        element2map[element] = False
                    trigger = element_parts[0].strip("[").strip()
                    event_type = element_parts[1].strip("]").strip()
                    if trigger.lower() not in lower_text:
                        element2map[element] = False
                    if event_type not in std_event_type:
                        element2map[element] = False
                elif len(element_parts) == 3:
                    span_text = element_parts[0].strip("[").strip()
                    entity_type = element_parts[1].strip()
                    try:
                        argument_type, target_word = element_parts[2].strip("]").strip().split("=")
                    except:
                        split_events = set([item.replace(element, "") for item in split_events])
                        element2map[element] = False
                        continue
                    argument_type, target_word = argument_type.strip(), target_word.strip()
                    if span_text.lower() not in lower_text or target_word.lower() not in lower_text:
                        element2map[element] = False
                    if entity_type not in mix_types:
                        element2map[element] = False
                    if argument_type not in std_argument_type:
                        element2map[element] = False
                else:
                    split_events = set([item.replace(element, "") for item in split_events])
                    element2map[element] = False
                    continue

            updated_events = []
            for event in split_events:
                flag = True
                for key, value in element2map.items():
                    if key in event and not value:
                        flag = False
                        break
                if flag:
                    result = event.split("]")[0]
                    result_split = result.split("|")
                    if len(result_split) == 1:
                        continue
                    if len(result_split) > 2:
                        event = event.replace(result, f"{result_split[0]}|{result_split[1]}").replace(" ]", "]")
                    assert len(event.split("]")[0].split("|")) == 2
                    event_type = event.split("]")[0].split("|")[1].strip()
                    # event_str = f"{event}->|{global_sent_start}-{global_sent_start + len(text)}|"
                    event_str = event
                    if event_str not in updated_events and event_type in std_event_type:
                        updated_events.append(event_str)

            segments = [k for k, v in element2map.items() if v]
            pos_map = set()
            for segment in segments:
                seg_parts = segment.split(" | ")
                if len(seg_parts) == 2:
                    trigger = seg_parts[0].strip("[").strip()
                    event_type = seg_parts[1].strip("]").strip()
                    pos_map.add((segment, trigger, event_type, "Event"))
                elif len(seg_parts) == 3:
                    argument = seg_parts[0].strip("[").strip()
                    argument_type = seg_parts[1].strip()
                    argument_role = seg_parts[2].strip("]").strip()
                    argument2span = argument_role.split("=")[-1].strip("]").strip()
                    argument2span_role = argument_role.split("=")[0].strip("]").strip()
                    if argument_type in std_event_type:
                        pos_map.add((f"[{argument} | {argument_type}", argument, argument_type, "Event"))
                    elif argument_type in std_entity_type:
                        pos_map.add((argument_role, argument2span, argument2span_role, "Argument"))
                        pos_map.add((f"[{argument}", argument, argument_type, "Argument"))
                    else:
                        raise ("error!")

                else:
                    raise ("error!")

            '''
            从UMLS映射和识别的实体当中给触发词和要素关联上CUI和TUI
            '''
            event_map = set()
            for (key, span, type_, indicator) in pos_map:
                if type_ not in std_event_type and type_ not in std_argument_type:
                    candidates = set()
                    select_terms = list()
                    for ent_i, ent in enumerate(meta_map):
                        sp_text, sp_type, CUIs, TUIs = ent
                        sp_text_ = sp_text.split(":##")[1].strip("##")
                        common_str = getNumofCommonSubstr(str1=span.lower(), str2=sp_text_.lower())
                        p = common_str[1] / (len(span) + 1e-10)
                        r = common_str[1] / (len(sp_text_) + 1e-10)
                        f = 2 * p * r / (p + r + 1e-10)
                        if f > 0.5:
                            select_terms.append([f, sp_text, CUIs, TUIs])

                    select_terms_ = sorted(select_terms, key=lambda x: x[0], reverse=True)
                    if len(select_terms_) > 0:
                        candidates.add(select_terms_[0][1])
                        umls_CUIs = select_terms_[0][2].split("|")
                        umls_TUIs = select_terms_[0][3].split("|")

                        if len(umls_CUIs) == 0:
                            umls_CUIs_str = "NA"
                            umls_TUIs_str = "NA"
                            candidates_str = "NA"
                        else:
                            umls_CUIs_str = "|".join(umls_CUIs)
                            umls_TUIs_str = "|".join(umls_TUIs)
                            candidates_str = "|".join(candidates)

                        temp = (key, span, type_, umls_CUIs_str, umls_TUIs_str, candidates_str, indicator)
                    else:
                        temp = (key, span, type_, "NA", "NA", "NA", indicator)
                    # print(span)
                else:
                    candidates = set()
                    select_terms = list()
                    for ent_i, ent in enumerate(terms):
                        start_, end_, sp_text, sp_type, CUIs, TUIs = ent.split("\t")
                        CUIs = CUIs.split("|")
                        TUIs = TUIs.split("|")
                        common_str = getNumofCommonSubstr(str1=span.lower(), str2=sp_text.lower())
                        p = common_str[1] / (len(span) + 1e-10)
                        r = common_str[1] / (len(sp_text) + 1e-10)
                        f = 2 * p * r / (p + r + 1e-10)
                        if f > 0.5:
                            select_terms.append(
                                (f, f"umls:##{sp_text}##",
                                 CUIs, TUIs))
                    select_terms_ = sorted(select_terms, key=lambda x: x[0], reverse=True)

                    if len(select_terms_) > 0:
                        candidates.add(select_terms_[0][1])
                        umls_CUIs, umls_TUIs = set(), set()
                        for CUI, TUI in zip(select_terms_[0][2], select_terms_[0][3]):
                            umls_CUIs.add(CUI)
                            umls_TUIs.add(TUI)

                        if len(umls_CUIs) == 0:
                            umls_CUIs_str = "NA"
                            umls_TUIs_str = "NA"
                            candidates_str = "NA"
                        else:
                            umls_CUIs_str = "|".join(list(umls_CUIs))
                            umls_TUIs_str = "|".join(list(umls_TUIs))
                            candidates_str = "|".join(list(candidates))

                        temp = (key, span, type_, umls_CUIs_str, umls_TUIs_str, candidates_str, indicator)
                    else:
                        temp = (key, span, type_, "NA", "NA", "NA", indicator)
                event_map.add(temp)

            '''统计事件和实体'''
            for k, updated_event in enumerate(updated_events):
                event_type = updated_event.split("]")[0].split("|")[1].strip()
                pure_event = updated_event
                if pure_event not in total_events and type(pure_event) == str:
                    event_count[event_type] += 1
                    total_events.add(pure_event)
                map_count = 0
                for evt_map in event_map:
                    if evt_map[0] in updated_event and evt_map[3] != "NA":
                        map_count += 1
                if map_count not in event_map_sata:
                    event_map_sata[map_count] = 0
                event_map_sata[map_count] += 1
            for meta in meta_map:
                if meta[0].startswith("scispacy"):
                    if meta[0].split(":##")[1].strip("##") not in total_entities and type(meta[0]) == str:
                        entity_count[meta[1]] += 1
                        total_entities.add(meta[0].split(":##")[1].strip("##"))
            for meta in event_map:
                if meta[2] not in std_entity_type:
                    continue
                total_entities.add(meta[0].strip().strip("]").strip("["))

            total_data["id"].append(info)
            total_data["text"].append(text)
            total_data["event"].append("\n".join(updated_events))
            total_data["mix_map"].append("\n".join(["\t".join(item) for item in event_map]))
            total_data["meta"].append("\n".join(["\t".join(item) for item in meta_map]))
            sent_num += 1
        except:
            continue

data = pd.DataFrame(total_data)
data.to_csv("./cache/event.csv", index=0, sep="\t")

os.makedirs("./cache/stata", exist_ok=True)

print(f"no_NA_sent: {no_NA_sent}, NA_sent: {NA_sent}, NA_rate: {round(NA_sent / (NA_sent + no_NA_sent), 4)}")

data = {"event": list(std_event_type), "argument": list(std_argument_type), "entity": list(std_entity_type)}

with open("./cache/stata/map.json", encoding="utf-8", mode="w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

with open("./cache/stata/event.txt", encoding="utf-8", mode="w") as f:
    for line in total_events:
        f.write(line + "\n")

with open("./cache/stata/entity.txt", encoding="utf-8", mode="w") as f:
    for line in total_entities:
        f.write(line + "\n")

statistics_data = {
    "总句子数(涵盖+不含)": no_NA_sent + NA_sent,
    "涵盖事件句子数": no_NA_sent,
    "不含事件句子数": NA_sent,
    "事件句子涵盖率": round(NA_sent / (NA_sent + no_NA_sent), 4),
    "非异常实句子总数": sent_num,
    "不重复事件总数": len(total_events),
    "不重复实体总数": len(total_entities),
    "不重复事件分布": event_count,
    "不重复实体分布": entity_count,
    "独立事件映射分布": event_map_sata}

# 事件数: 160, 要素数: 15, 实体数: 140

with open("./cache/stata/stat.json", encoding="utf-8", mode="w") as f:
    json.dump(statistics_data, f, indent=4, ensure_ascii=False)
