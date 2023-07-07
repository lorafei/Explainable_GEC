import json
from tqdm.auto import tqdm
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default='data/json/train.json')
parser.add_argument("--save_file", type=str, default='data/ner/train.pkl')
args = parser.parse_args()


sep_token = ['[MOD]']
data = []

with open(args.data_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

def idx2vec(idx, length):
    if isinstance(idx, list):
        if len(idx) != 0:
            idx_vec = [1 if i in idx else 0 for i in range(length)]
        else:
            idx_vec = [0] * length
    elif isinstance(idx, dict):
        idx_vec = [0] * length
        for k, v in idx.items():
            if int(k) < len(idx_vec):
                idx_vec[int(k)] = v
    return idx_vec

classes_ids = {
    "Infinitives": "s1",
    "Gerund": "s2",
    "Participle": "s3",
    "Subject-Verb Agreement": "s5",
    "Auxiliary Verb": "s6",
    "Verb Tense": "s7",
    "Pronoun-Antecedent Agreement": "s8",
    "Possessive": "s9",
    "Collocation": "m1",
    "Preposition": "m2",
    "POS Confusion": "m3",
    "Article": "m4",
    "Number": "m5",
    "Transitive Verb": "m6",
    "Others": "oth"
}

# convert into ner data structure
def sent_to_ner_data(sentence, evidence_index, correction_index, error_type, parsing_order, ids=0, prefix=True):
    sent_data = []
    for j, token in enumerate(sentence):
        new_tokens = token
        if evidence_index:  # for dev and train data
            if evidence_index[j]:
                type = classes_ids[error_type]
                if prefix:
                    if j == 0 or evidence_index[j - 1] == 0:
                        cls = 'B-' + type
                    else:
                        cls = 'I-' + type
                else:
                    cls = type
            else:
                cls = 'O'
            sent_data.append([
                ids,
                new_tokens,
                cls,
                classes_ids[error_type],
                correction_index[j],
                parsing_order[j]
            ])
    return sent_data


new_data = []
for i, d in tqdm(enumerate(data)):
    sentence = d['target'] + sep_token + d['source']
    correction_index = idx2vec(d['correction_index'], len(sentence))
    evidence_index = idx2vec(d['evidence_index'], len(sentence))
    predicted_parsing_order = idx2vec(d['predicted_parsing_order'], len(sentence))
    ner_d = sent_to_ner_data(sentence, evidence_index, correction_index, d['error_type'], predicted_parsing_order, ids=i)
    new_data.extend(ner_d)

with open(args.save_file, 'wb') as f:
    pickle.dump(new_data, f)
