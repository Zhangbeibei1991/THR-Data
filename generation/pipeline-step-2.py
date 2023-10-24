from tqdm import tqdm
import json
import os
import spacy
import pandas as pd

nlp = spacy.load("en_core_sci_lg")
os.makedirs("sentence", exist_ok=True)
sent_num, doc_num, save_num = 0, 0, 0
with open("cache/pubmed.txt", encoding="utf-8", mode="r") as f:
    batch_size, batch_count = 10000, 0
    csv_dict = {"info": [], "sentence": [], "text_label": []}
    for k, line in tqdm(enumerate(f.readlines()), desc="running"):
        line_dict = json.loads(line.strip())
        ID = line_dict["id"]
        doc_num += 1
        context = line_dict["text"]
        # context = context.replace(")", " ) ").replace("(", " ( ")
        # context = context.replace("]", " ] ").replace("[", " [ ")
        # context = context.replace("-", " - ")
        context = context.replace("  ", " ")
        spacy_context0 = nlp(context)
        if k == 0:
            first_id = ID
        batch_count += 1
        if batch_count > 0 and batch_count % batch_size == 0:
            batch_sent_num = len(csv_dict["info"])
            print(f"save to ./sentence/PMID-{first_id}-To-PMID{last_id}-has-{batch_sent_num}.tsv")
            pd.DataFrame(csv_dict).to_csv(f"./sentence/PMID-{first_id}-To-PMID-{last_id}-has-{batch_sent_num}.tsv", sep="\t", index=0)
            csv_dict = {"info": [], "sentence": [], "text_label": []}
            first_id = ID
            save_num += 1
        for sent in spacy_context0.sents:
            csv_dict["info"].append(f"PMID-{ID}.s:" + "@".join([str(sent.start_char), str(sent.end_char)]))
            csv_dict["sentence"].append(sent.text)
            csv_dict["text_label"].append("NA")
            sent_num += 1
        last_id = ID
batch_sent_num = len(csv_dict["info"])
pd.DataFrame(csv_dict).to_csv(f"./sentence/PMID-{first_id}-To-PMID-{last_id}-has-{batch_sent_num}.tsv", sep="\t", index=0)
print(f"save num: {save_num + 1}")
print(f"sent num: {sent_num}/doc num: {doc_num} == {round(sent_num / doc_num)}")


'''
save num: 6
sent num: 5887896/doc num: 570035 == 10
'''