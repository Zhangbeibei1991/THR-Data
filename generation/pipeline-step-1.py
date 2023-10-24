'''
In the first stage, about 20w+ abstracts are selected from the downloaded pubmed abstract set
according the number (threshold) of co-occurrence of trigger collected from mlee dataset.
Here the threshold of co-occurrence is set as 30. In addition, to obtain the more related and more
structuring abstracts, we also require that the selected abstracts must contain the keyword of "experiment"
'''
import os
import pandas as pd
import numpy as np
import json


class Document:
    """"""

    def __init__(self, doc_id, paragraphs, mesh_terms, chemical_list, overlap_list):
        self.doc_id = doc_id
        self.paragraphs = paragraphs
        self.mesh_terms = mesh_terms
        self.chemical_list = chemical_list
        self.overlap_list = overlap_list

    def __str__(self):
        return "{}:({},{},{})".format(
            self.doc_id,
            [p for p in self.paragraphs],
            [p for p in self.mesh_terms],
            [p for p in self.chemical_list])


# root_path = '/media/linus/00025B030003677A/PubMed/journal_to_csv'
root_path = 'H:\PubMed\journal_to_csv'
file_names = os.listdir(root_path)
save_path = 'cache'
os.makedirs(save_path, exist_ok=True)
# print(file_names)

threshold = 1


def parse_document_files(titles, abstracts, doc_ids, mesh_terms, chemical_lists, mode='experiment'):
    documemts = []
    file_path = os.path.join(save_path, 'pubmed.txt')
    f = open(file_path, encoding='utf-8', mode='a')
    for doc_id, title, abstract, mesh_term, chemical_list in zip(doc_ids, titles, abstracts, mesh_terms,
                                                                 chemical_lists):
        # whether the text contains key word "experiment"
        if len(title) > 0 and title.strip()[-1] not in [".", "!", "?"]:
            title += "."
        text = f"{title} {abstract}".replace("\n", " ")

        # 判定文档中蛋白质的数目
        protein_count = text.lower().count("protein")
        if protein_count < threshold: continue
        if not ("cancer" in text.lower() or "tumor" in text.lower()
                or "melanoma" in text.lower() or "neoplams" in text.lower()
                or "tumour" in text.lower() or "carcinoma" in text.lower()
                or "leukemia" in text.lower() or "leucocythemia" in text.lower()
                or "adenocarcinoma" in text.lower() or "hepatocarcinoma" in text.lower()
                or "oncogenesis" in text.lower()):
            continue
        document = Document(paragraphs=[title, abstract], doc_id=f'PMID-{doc_id}', mesh_terms=mesh_term,
                            chemical_list=chemical_list, overlap_list=None)
        documemts.append(document)
        # text = {'title': title, 'abstract': abstract}
        example_dicts = {'id': doc_id, 'text': text}
        f.write(json.dumps(example_dicts) + '\n')
    f.close()
    return documemts


count = 0
total = 0
for k, file_name in enumerate(file_names):
    file_path = os.path.join(root_path, file_name)
    data = pd.read_csv(file_path).replace(np.NAN, '')
    doc_ids = data['pmid'].tolist()
    titles = data['title'].tolist()
    abstracts = data['abstract'].tolist()
    mesh_terms = data['mesh_terms'].tolist()
    chemical_lists = data['chemical_list'].tolist()
    documemts = parse_document_files(titles, abstracts, doc_ids, mesh_terms, chemical_lists, mode='experiment')
    # create_annotations(documents=documemts)
    count += len(documemts)
    total += len(titles)
    print(f'{file_name}: {total}, {len(documemts)}/{count} for {k}/{len(file_names)}')

# 5 => pubmed21n1061.csv: 21073378, 363/504974 for 1056/1057
# 10 => pubmed21n1061.csv: 21073378, 46/73746 for 1056/1057
