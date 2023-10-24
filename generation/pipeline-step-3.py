import spacy
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from scispacy.linking import EntityLinker

nlp = spacy.load("en_core_sci_lg")

nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

nlp1 = spacy.load("en_ner_jnlpba_md")
nlp2 = spacy.load("en_ner_bionlp13cg_md")

sent_num = 0


def my_softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def clean(spacy_context1, text, start_token, end_token):
    left_count = text.count("(")
    right_count = text.count(")")

    if left_count > right_count:
        temp_end, count = end_token, 0
        for m, k_char in enumerate(spacy_context1[end_token:]):
            if k_char.text == ")":
                temp_end = end_token + m + 1
                count += 1
            if count == left_count - right_count:
                break
        end_token = temp_end
        text = spacy_context1[start_token:end_token].text

    if left_count < right_count:
        temp_start, count = start_token, 0
        for m, k_char in enumerate(reversed(spacy_context1[:start_token])):
            if k_char.text == "(":
                temp_start = start_token - m - 1
                count += 1
            if count == right_count - left_count:
                break
        start_token = temp_start
        text = spacy_context1[start_token:end_token].text

        for m, item in enumerate(reversed(spacy_context1[:start_token])):
            if (item.tag_ in ["IN", "TO", ",", "(", ".", "["] or item.text in ["as", "with", "by"]) and m <= 5:
                start_token = start_token - m
                if spacy_context1[start_token].text.lower() == "the":
                    start_token += 1
                text = spacy_context1[start_token:end_token].text
                break

    left_count = text.count("[")
    right_count = text.count("]")
    if left_count > right_count:
        temp_end, count = end_token, 0
        for m, k_char in enumerate(spacy_context1[end_token:]):
            if k_char.text == "]":
                temp_end = end_token + m + 1
                count += 1
            if count == left_count - right_count:
                break
        end_token = temp_end
        text = spacy_context1[start_token:end_token].text

    if left_count < right_count:
        temp_start, count = start_token, 0
        for m, k_char in enumerate(reversed(spacy_context1[:start_token])):
            if k_char.text == "[":
                temp_start = start_token - m - 1
                count += 1
            if count == right_count - left_count:
                break
        start_token = temp_start
        text = spacy_context1[start_token:end_token].text

        for m, item in enumerate(reversed(spacy_context1[:start_token])):
            if (item.tag_ in ["IN", "TO", ",", "(", ".", "["] or item.text in ["as", "with", "by"]) and m <= 5:
                start_token = start_token - m
                if spacy_context1[start_token].text.lower() == "the":
                    start_token += 1
                text = spacy_context1[start_token:end_token].text
                break
    return text, start_token, end_token, label


data_names = [item for item in os.listdir("./sentence") if "-Entity" not in item and "-has" in item and "Event" not in item]
data_dir = "./sentence"
for data_name in data_names:
    data = pd.read_csv(os.path.join(data_dir, data_name), sep="\t")
    data["entity_label"] = ["ND" for _ in range(len(data))]
    data["item_label"] = ["ND" for _ in range(len(data))]
    for t, (info, context, text_label, entity_label, _) in tqdm(enumerate(data.values.tolist()), desc=data_name):
        spacy_context0 = nlp(context)
        spacy_context1 = nlp1(context)
        spacy_context2 = nlp2(context)

        umls_ents = []
        for ent_span in spacy_context0.ents:
            start_char, end_char, text, label = ent_span.start_char, ent_span.end_char, ent_span.text, ent_span.label_
            for umls_ent in ent_span._.kb_ents:
                target = linker.kb.cui_to_entity[umls_ent[0]]
                cui_id = target.concept_id
                tuis = "@".join(target.types)
                umls_ents.append([start_char, end_char, text, label, cui_id, tuis])
                break

        tokens, entities1, entities2 = [], [], []
        for token in spacy_context1:
            tokens.append(token.text)
        spans_start, spans_end, spans, text_set = set(), set(), set(), set()
        for ent_span in spacy_context1.ents:
            start_char, end_char, text, label = ent_span.start_char, ent_span.end_char, ent_span.text, ent_span.label_
            target_set = set(list(range(start_char, end_char)))
            cur_cuis, cur_tuis = [], []
            for item in umls_ents:
                umls_start_char, umls_end_char, _, _, cui_id, tuis = item
                umls_set = set(list(range(umls_start_char, umls_end_char)))
                if len(target_set & umls_set) > 0:
                    cur_cuis.append(cui_id)
                    cur_tuis.append(tuis)

            cur_cuis = "|".join(cur_cuis)
            cur_tuis = "|".join(cur_tuis)

            start_token, end_token = ent_span.start, ent_span.end
            spans_start.add(start_token)
            spans_end.add(end_token)
            spans.add((start_token, end_token))
            text, start_token, end_token, _, = clean(spacy_context1, text, start_token, end_token)
            text, start_token, end_token, _, = clean(spacy_context1, text, start_token, end_token)

            if "http" in text:
                continue

            if label in ["PROTEIN", "DNA", "RNA"]:
                label = "GENE_OR_GENE_PRODUCT"

            # 可以用下面这种span的方法得到词的字符串位置
            text = spacy_context1[start_token:end_token].text
            start_char = spacy_context1[start_token:end_token].start_char
            end_char = spacy_context1[start_token:end_token].end_char

            text_set.add(text)

            entities1.append([text, label, start_token, end_token, start_char, end_char, cur_cuis, cur_tuis])

        for ent_span in spacy_context2.ents:
            start_char, end_char, text, label = ent_span.start_char, ent_span.end_char, ent_span.text, ent_span.label_
            start_token, end_token = ent_span.start, ent_span.end
            label1 = label

            target_set = set(list(range(start_char, end_char)))
            cur_cuis, cur_tuis = [], []
            for item in umls_ents:
                umls_start_char, umls_end_char, _, _, cui_id, tuis = item
                umls_set = set(list(range(umls_start_char, umls_end_char)))
                if len(target_set & umls_set) > 0:
                    cur_cuis.append(cui_id)
                    cur_tuis.append(tuis)

            cur_cuis = "|".join(cur_cuis)
            cur_tuis = "|".join(cur_tuis)

            if start_token in spans_start or end_token in spans_end:
                continue
            flag = False
            for s, e in spans:
                if s <= start_token <= end_token <= e:
                    flag = True
                    break
            if flag:
                continue
            text, start_token, end_token, _, = clean(spacy_context2, text, start_token, end_token)
            text, start_token, end_token, _, = clean(spacy_context2, text, start_token, end_token)
            text = spacy_context1[start_token: end_token].text

            start_char = spacy_context1[start_token:end_token].start_char
            end_char = spacy_context1[start_token:end_token].end_char

            if "http" in text:
                continue
            if "protein" in text.lower() or text in text_set:
                label = "GENE_OR_GENE_PRODUCT"
                entities1.append([text, label, start_token, end_token, start_char, end_char, cur_cuis, cur_tuis])
            elif text.lower().endswith("mrna") or text.lower().endswith("rna"):
                label = "GENE_OR_GENE_PRODUCT"
                entities1.append([text, label, start_token, end_token, start_char, end_char, cur_cuis, cur_tuis])
            elif text.lower().endswith("receptor") or text.lower().endswith("receptors"):
                label = "GENE_OR_GENE_PRODUCT"
                entities1.append([text, label, start_token, end_token, start_char, end_char, cur_cuis, cur_tuis])
            elif text.lower().endswith("dna"):
                label = "GENE_OR_GENE_PRODUCT"
                entities1.append([text, label, start_token, end_token, start_char, end_char, cur_cuis, cur_tuis])
            elif text.lower().endswith("cell") or text.lower().endswith("cells"):
                label = "CELL_LINE"
                entities1.append([text, label, start_token, end_token, start_char, end_char, cur_cuis, cur_tuis])
            elif text.endswith(".") or "&" in text or "e.g." in text or "%" in text or ":" in text or "et al" in text:
                continue
            else:
                if " is " in text:
                    index = text.split().index("is")
                    end_token = start_token + index
                    text = spacy_context1[start_token:end_token].text

                    start_char = spacy_context1[start_token:end_token].start_char
                    end_char = spacy_context1[start_token:end_token].end_char

                if text.startswith("-"):
                    continue

                entities2.append([text, label, start_token, end_token, start_char, end_char, cur_cuis, cur_tuis])
        entities1 = sorted(entities1, key=lambda x: x[2])
        entities = entities1 + entities2
        doc_text = " ".join(tokens)
        entities = sorted(entities, key=lambda x: x[2])
        total_count = len(entities)
        match_count = sum([1 for item in entities if len(item[-1]) != 0])
        final_entities = []
        for k1, ent in enumerate(entities):
            _, label, start_token, end_token, start_char, end_char, cui, tui = ent
            if tokens[end_token - 1] in ["]", ")"] and tokens[start_token] not in ["[", "("]:
                end_token -= 1
            if tokens[end_token - 1] in [",", ":"]:
                end_token -= 1
            if tokens[end_token - 1] in ["or", "and"]:
                end_token -= 1
            ent_tokens = tokens[start_token:end_token]
            if ent_tokens.count("[") == 1 and ent_tokens.count("]") == 0:
                for k in range(end_token, len(tokens)):
                    if tokens[k] == "]":
                        end_token = k + 1
                        break
            if ent_tokens.count("(") == 1 and ent_tokens.count(")") == 0:
                for k in range(end_token, len(tokens)):
                    if tokens[k] == ")":
                        end_token = k + 1
                        break
            text = " ".join(tokens[start_token:end_token])
            if len(cui) == 0:
                cui = "NA"
                tui = "NA"
            if label == "CELL_TYPE" or label == "CELL":
                label = "CELL_LINE"

            start_char = spacy_context1[start_token:end_token].start_char
            end_char = spacy_context1[start_token:end_token].end_char

            item = f"{label}\t{start_char} {end_char}\t{text}\t{cui}\t{tui}"
            final_entities.append(item)
        data.loc[t, "entity_label"] = "$$".join(final_entities)
        data.loc[t, "item_label"] = "$$".join(["\t".join([str(item) for item in items]) for items in umls_ents])

    data.to_csv(os.path.join(data_dir, data_name.replace(".tsv", "-Entity.tsv")), sep="\t", index=0)
