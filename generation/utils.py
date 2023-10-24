import re
import pandas as pd

def filter_property(event_mention):
    signal_set = set()
    if "CL =" in event_mention:
        reses = re.finditer(r"\| CL = .+?\|", event_mention)
        for res in reses:
            char_start, char_end = res.start(), res.end()
            signal_set.add((char_start, char_end, event_mention[char_start:char_end]))
    if "KT =" in event_mention:
        reses = re.finditer(r"\| KT = .+?\|", event_mention)
        for res in reses:
            char_start, char_end = res.start(), res.end()
            signal_set.add((char_start, char_end, event_mention[char_start:char_end]))
    if "Manner =" in event_mention:
        reses = re.finditer(r"\| Manner = .+?(\||\])", event_mention)
        for res in reses:
            char_start, char_end = res.start(), res.end()
            signal_set.add((char_start, char_end, event_mention[char_start:char_end]))
    if "Polarity =" in event_mention:
        reses = re.finditer(r"\| Polarity = .+?(\||\])", event_mention)
        for res in reses:
            char_start, char_end = res.start(), res.end()
            signal_set.add((char_start, char_end, event_mention[char_start:char_end]))
    if "Source =" in event_mention:
        reses = re.finditer(r"\| Source = .+?(\||\])", event_mention)
        for res in reses:
            char_start, char_end = res.start(), res.end()
            signal_set.add((char_start, char_end, event_mention[char_start:char_end]))

    if "Speculation =" in event_mention:
        reses = re.finditer(r"\| Speculation = .+?(\||\])", event_mention)
        for res in reses:
            char_start, char_end = res.start(), res.end()
            signal_set.add((char_start, char_end, event_mention[char_start:char_end]))

    new_mention = ""
    for char_i, char_each in enumerate(event_mention):
        flag = True
        for (char_start, char_end, replace_part) in signal_set:
            if char_start <= char_i < char_end - 1:
                flag = False
                break
        if flag:
            new_mention += char_each
    new_mention = new_mention.replace(" ]", "]")
    return new_mention

def parse_each_event(event):
    std_event_type, std_argument_type, std_entity_type = set(), set(), set()
    # if 'Cue =' in event:
    #     print()
    if "=" in event:
        # argument_type = event.split("=")[0].split("|")[-1].strip()
        results = re.finditer("=", event)
        for result in results:
            char_start = result.start()
            org_start = char_start
            while True:
                if event[char_start] == "|":
                    break
                else:
                    char_start -= 1
            argument_type = event[char_start + 1:org_start].strip()
            last_char_start = char_start - 1
            while True:
                if event[last_char_start] == "|":
                    last_char_start += 1
                    break
                else:
                    last_char_start -= 1
            span_text = event[last_char_start:char_start].strip()

            std_argument_type.add(argument_type)
            std_entity_type.add(span_text)
    event_type = event.split("]")[0].split("|")[-1].strip()
    std_event_type.add(event_type)
    return std_event_type, std_argument_type, std_entity_type


def getNumofCommonSubstr(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
    # 开辟列表空间 为什么要多一位呢?主要是不多一位的话,会存在边界问题
    # 多了一位以后就不存在超界问题
    maxNum = 0  # 最长匹配长度
    p = 0  # 匹配的起始位

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    return str1[p - maxNum:p], maxNum


def get_std_set(root_path):
    # 读取标注数据集里面的事件类型、实体类型和要素类型
    train_biot2e = pd.read_csv(f"{root_path}/train.tsv", sep="\t").values.tolist()
    valid_biot2e = pd.read_csv(f"{root_path}/validation.tsv", sep="\t").values.tolist()
    test_biot2e = pd.read_csv(f"{root_path}/test.tsv", sep="\t").values.tolist()

    biot2e_data = train_biot2e + valid_biot2e + test_biot2e
    std_event_type, std_entity_type, std_argument_type = set(), set(), set()
    for s, line in enumerate(biot2e_data):
        event_mentions = line[-1]
        if event_mentions != "ND":
            events = event_mentions.split("   ")
            for event in events:
                event = filter_property(event)
                event_type, argument_type, entity_type = parse_each_event(event)
                std_event_type.update(event_type)
                std_argument_type.update(argument_type)
                std_entity_type.update(entity_type)
    std_entity_type = std_entity_type - std_event_type
    std_event_type = std_event_type - std_entity_type
    return std_event_type, std_argument_type, std_entity_type



class Stack:
    def __init__(self):
        self.items = []

    def pop_(self):
        return self.items[-1]

    def pop(self):
        return self.items.pop()

    def get_length(self):
        return len(self.items)

    def push(self, elem):
        return self.items.append(elem)


def Match_blankets(string):
    new_string = ""
    left_count, right_count = 0, 0
    for char_each in string:
        if char_each == "[":
            new_string += char_each
            left_count += 1
        elif char_each == "]":
            new_string += char_each
            right_count += 1
    if left_count != right_count:
        # print("括号数目不匹配！")
        return -1
    string = new_string
    stack = Stack()
    stack2 = Stack()
    match_dict = {")": "(", "]": "[", "}": "{"}
    for i in string:
        if i in {"(", "[", "{"}:
            stack.push(i)
        elif i in {")", "]", "}"}:
            get_ = stack.pop_()
            if get_ == match_dict[i]:
                stack.pop()
            else:
                stack2.push(i)
        else:
            # print("括号不纯！")
            return -1
    if stack.get_length() != stack2.get_length():
        # print("匹配失败！")
        return -1
    else:
        for i in stack2.items:
            get_new = stack.pop_()
            if get_new == match_dict[i]:
                stack.pop()
                stack2.pop()
        if stack.get_length() == 0 and stack2.get_length() == 0:
            # print("匹配成功！")
            return 0


def iter_position(data, key_num):
    if key_num == 1:
        position_list = []
        for map_item in data[0]:
            key, char_pos = map_item.split("@")
            ans = [f"{key}@{char_pos}"]
            ans = list(set(ans))
            if ans not in position_list:
                position_list.append(ans)
    elif key_num == 2:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}"]
                ans = list(set(ans))
                if ans not in position_list:
                    position_list.append(ans)
    elif key_num == 3:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}", f"{key3}@{char_pos3}"]
                    ans = list(set(ans))
                    if ans not in position_list:
                        position_list.append(ans)
    elif key_num == 4:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}", f"{key3}@{char_pos3}",
                               f"{key4}@{char_pos4}"]
                        ans = list(set(ans))
                        if ans not in position_list:
                            position_list.append(ans)

    elif key_num == 5:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}", f"{key3}@{char_pos3}",
                                   f"{key4}@{char_pos4}", f"{key5}@{char_pos5}"]
                            ans = list(set(ans))
                            if ans not in position_list:
                                position_list.append(ans)

    elif key_num == 6:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            for map_item6 in data[5]:
                                key6, char_pos6 = map_item6.split("@")
                                ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}", f"{key3}@{char_pos3}",
                                       f"{key4}@{char_pos4}", f"{key5}@{char_pos5}", f"{key6}@{char_pos6}"]
                                ans = list(set(ans))
                                if ans not in position_list:
                                    position_list.append(ans)
    elif key_num == 7:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            for map_item6 in data[5]:
                                key6, char_pos6 = map_item6.split("@")
                                for map_item7 in data[6]:
                                    key7, char_pos7 = map_item7.split("@")
                                    ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}", f"{key3}@{char_pos3}",
                                           f"{key4}@{char_pos4}", f"{key5}@{char_pos5}", f"{key6}@{char_pos6}",
                                           f"{key7}@{char_pos7}"]
                                    ans = list(set(ans))
                                    if ans not in position_list:
                                        position_list.append(ans)

    elif key_num == 8:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            for map_item6 in data[5]:
                                key6, char_pos6 = map_item6.split("@")
                                for map_item7 in data[6]:
                                    key7, char_pos7 = map_item7.split("@")
                                    for map_item8 in data[7]:
                                        key8, char_pos8 = map_item8.split("@")
                                        ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}", f"{key3}@{char_pos3}",
                                               f"{key4}@{char_pos4}", f"{key5}@{char_pos5}", f"{key6}@{char_pos6}",
                                               f"{key7}@{char_pos7}", f"{key8}@{char_pos8}"]
                                        ans = list(set(ans))
                                        if ans not in position_list:
                                            position_list.append(ans)
    elif key_num == 9:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            for map_item6 in data[5]:
                                key6, char_pos6 = map_item6.split("@")
                                for map_item7 in data[6]:
                                    key7, char_pos7 = map_item7.split("@")
                                    for map_item8 in data[7]:
                                        key8, char_pos8 = map_item8.split("@")
                                        for map_item9 in data[8]:
                                            key9, char_pos9 = map_item9.split("@")
                                            ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}", f"{key3}@{char_pos3}",
                                                   f"{key4}@{char_pos4}", f"{key5}@{char_pos5}", f"{key6}@{char_pos6}",
                                                   f"{key7}@{char_pos7}", f"{key8}@{char_pos8}", f"{key9}@{char_pos9}"]
                                            ans = list(set(ans))
                                            if ans not in position_list:
                                                position_list.append(ans)

    elif key_num == 10:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            for map_item6 in data[5]:
                                key6, char_pos6 = map_item6.split("@")
                                for map_item7 in data[6]:
                                    key7, char_pos7 = map_item7.split("@")
                                    for map_item8 in data[7]:
                                        key8, char_pos8 = map_item8.split("@")
                                        for map_item9 in data[8]:
                                            key9, char_pos9 = map_item9.split("@")
                                            for map_item10 in data[9]:
                                                key10, char_pos10 = map_item10.split("@")
                                                ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}",
                                                       f"{key3}@{char_pos3}",
                                                       f"{key4}@{char_pos4}", f"{key5}@{char_pos5}",
                                                       f"{key6}@{char_pos6}",
                                                       f"{key7}@{char_pos7}", f"{key8}@{char_pos8}",
                                                       f"{key9}@{char_pos9}",
                                                       f"{key10}@{char_pos10}"]
                                                ans = list(set(ans))
                                                if ans not in position_list:
                                                    position_list.append(ans)
    elif key_num == 11:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            for map_item6 in data[5]:
                                key6, char_pos6 = map_item6.split("@")
                                for map_item7 in data[6]:
                                    key7, char_pos7 = map_item7.split("@")
                                    for map_item8 in data[7]:
                                        key8, char_pos8 = map_item8.split("@")
                                        for map_item9 in data[8]:
                                            key9, char_pos9 = map_item9.split("@")
                                            for map_item10 in data[9]:
                                                key10, char_pos10 = map_item10.split("@")
                                                for map_item11 in data[10]:
                                                    key11, char_pos11 = map_item11.split("@")
                                                    ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}",
                                                           f"{key3}@{char_pos3}",
                                                           f"{key4}@{char_pos4}", f"{key5}@{char_pos5}",
                                                           f"{key6}@{char_pos6}",
                                                           f"{key7}@{char_pos7}", f"{key8}@{char_pos8}",
                                                           f"{key9}@{char_pos9}",
                                                           f"{key10}@{char_pos10}", f"{key11}@{char_pos11}"]
                                                    ans = list(set(ans))
                                                    if ans not in position_list:
                                                        position_list.append(ans)

    elif key_num == 12:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            for map_item6 in data[5]:
                                key6, char_pos6 = map_item6.split("@")
                                for map_item7 in data[6]:
                                    key7, char_pos7 = map_item7.split("@")
                                    for map_item8 in data[7]:
                                        key8, char_pos8 = map_item8.split("@")
                                        for map_item9 in data[8]:
                                            key9, char_pos9 = map_item9.split("@")
                                            for map_item10 in data[9]:
                                                key10, char_pos10 = map_item10.split("@")
                                                for map_item11 in data[10]:
                                                    key11, char_pos11 = map_item11.split("@")
                                                    for map_item12 in data[11]:
                                                        key12, char_pos12 = map_item12.split("@")
                                                        ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}",
                                                               f"{key3}@{char_pos3}",
                                                               f"{key4}@{char_pos4}", f"{key5}@{char_pos5}",
                                                               f"{key6}@{char_pos6}",
                                                               f"{key7}@{char_pos7}", f"{key8}@{char_pos8}",
                                                               f"{key9}@{char_pos9}",
                                                               f"{key10}@{char_pos10}", f"{key11}@{char_pos11}",
                                                               f"{key12}@{char_pos12}"]
                                                        ans = list(set(ans))
                                                        if ans not in position_list:
                                                            position_list.append(ans)
    elif key_num == 13:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            for map_item6 in data[5]:
                                key6, char_pos6 = map_item6.split("@")
                                for map_item7 in data[6]:
                                    key7, char_pos7 = map_item7.split("@")
                                    for map_item8 in data[7]:
                                        key8, char_pos8 = map_item8.split("@")
                                        for map_item9 in data[8]:
                                            key9, char_pos9 = map_item9.split("@")
                                            for map_item10 in data[9]:
                                                key10, char_pos10 = map_item10.split("@")
                                                for map_item11 in data[10]:
                                                    key11, char_pos11 = map_item11.split("@")
                                                    for map_item12 in data[11]:
                                                        key12, char_pos12 = map_item12.split("@")
                                                        for map_item13 in data[12]:
                                                            key13, char_pos13 = map_item13.split("@")
                                                            ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}",
                                                                   f"{key3}@{char_pos3}",
                                                                   f"{key4}@{char_pos4}", f"{key5}@{char_pos5}",
                                                                   f"{key6}@{char_pos6}",
                                                                   f"{key7}@{char_pos7}", f"{key8}@{char_pos8}",
                                                                   f"{key9}@{char_pos9}",
                                                                   f"{key10}@{char_pos10}", f"{key11}@{char_pos11}",
                                                                   f"{key12}@{char_pos12}",
                                                                   f"{key13}@{char_pos13}"]
                                                            ans = list(set(ans))
                                                            if ans not in position_list:
                                                                position_list.append(ans)
    elif key_num == 14:
        position_list = []
        for map_item1 in data[0]:
            key1, char_pos1 = map_item1.split("@")
            for map_item2 in data[1]:
                key2, char_pos2 = map_item2.split("@")
                for map_item3 in data[2]:
                    key3, char_pos3 = map_item3.split("@")
                    for map_item4 in data[3]:
                        key4, char_pos4 = map_item4.split("@")
                        for map_item5 in data[4]:
                            key5, char_pos5 = map_item5.split("@")
                            for map_item6 in data[5]:
                                key6, char_pos6 = map_item6.split("@")
                                for map_item7 in data[6]:
                                    key7, char_pos7 = map_item7.split("@")
                                    for map_item8 in data[7]:
                                        key8, char_pos8 = map_item8.split("@")
                                        for map_item9 in data[8]:
                                            key9, char_pos9 = map_item9.split("@")
                                            for map_item10 in data[9]:
                                                key10, char_pos10 = map_item10.split("@")
                                                for map_item11 in data[10]:
                                                    key11, char_pos11 = map_item11.split("@")
                                                    for map_item12 in data[11]:
                                                        key12, char_pos12 = map_item12.split("@")
                                                        for map_item13 in data[12]:
                                                            key13, char_pos13 = map_item13.split("@")
                                                            for map_item14 in data[13]:
                                                                key14, char_pos14 = map_item14.split("@")
                                                                ans = [f"{key1}@{char_pos1}", f"{key2}@{char_pos2}",
                                                                       f"{key3}@{char_pos3}",
                                                                       f"{key4}@{char_pos4}", f"{key5}@{char_pos5}",
                                                                       f"{key6}@{char_pos6}",
                                                                       f"{key7}@{char_pos7}", f"{key8}@{char_pos8}",
                                                                       f"{key9}@{char_pos9}",
                                                                       f"{key10}@{char_pos10}", f"{key11}@{char_pos11}",
                                                                       f"{key12}@{char_pos12}",
                                                                       f"{key13}@{char_pos13}", f"{key14}@{char_pos14}"]
                                                                ans = list(set(ans))
                                                                if ans not in position_list:
                                                                    position_list.append(ans)
    else:
        raise ("error!")
    return position_list