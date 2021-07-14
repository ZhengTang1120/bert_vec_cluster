from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

from sklearn.cluster import AffinityPropagation

import nltk

from os import walk
import json
from collections import defaultdict


def get_vectors(sentences):
    vectors = list()
    words = list()
    for text in sentences:
        text = text.lower()
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        for i, token in enumerate(nltk.pos_tag(tokenizer.convert_ids_to_tokens(encoded_input.input_ids.data.numpy().tolist()[0]))):
            if token[-1].startswith('VB') and token[0] not in ['[CLS]','[SEP]']:
                words.append(token[0])
                vectors.append(output.last_hidden_state[0].data.numpy()[i])
            
    return words, vectors

def cluster(vectors):
    af = AffinityPropagation(preference=-50).fit(vectors)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    return labels


data_path = "/Users/zheng/Downloads/100cdr/"
filenames = next(walk(data_path), (None, None, []))[2][:5]
sentences = list()
for file in filenames:
    if file.endswith('.json'):
        for annotation in json.load(open(data_path+file))['annotations']:
            if annotation['label'] == 'qntfy-key-sentence-annotator':
                for content in annotation['content']:
                    sentences += [content['value']]
words, vectors = get_vectors(sentences)
cluster_ids = cluster(vectors)
clusters = defaultdict(list)
for i, c_id in enumerate(cluster_ids):
    clusters[int(c_id)].append(words[i])
print (json.dumps(clusters))






