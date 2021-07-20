from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

from sklearn.cluster import MeanShift, AffinityPropagation, KMeans, estimate_bandwidth

import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from os import walk
import json
from collections import defaultdict
from random import sample


def get_vectors(sentences):
    vectors = defaultdict(list)
    for text in sentences:
        text = text.lower()
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        for i, token in enumerate(nltk.pos_tag(tokenizer.convert_ids_to_tokens(encoded_input.input_ids.data.numpy().tolist()[0]))):
            if token[-1].startswith('VB') and token[0] not in ['[CLS]','[SEP]', '/', '[', ']', '\\'] and wnl.lemmatize(token[0], pos='v') not in ['get', 'be', 'have']:
                vectors[token[0]].append(output.last_hidden_state[0].data.numpy()[i])
            
    return vectors

def cluster_af(vectors):

    af = AffinityPropagation(preference=-50).fit(vectors)
    cluster_centers = af.cluster_centers_
    labels = af.labels_

    return cluster_centers, labels

def cluster_km(vectors, n):

    km = KMeans(n_clusters=min(n, len(vectors)), random_state=170).fit(vectors)
    labels = km.labels_
    cluster_centers = km.cluster_centers_

    return cluster_centers, labels

def cluster_ms(vectors):
    if len(vectors) == 1:
        return vectors, [0]
    bandwidth = estimate_bandwidth(vectors, quantile=0.5, n_samples=min(len(vectors), 500))
    bandwidth = bandwidth if bandwidth!=0 else None
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(vectors)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_


    return cluster_centers, labels


data_path = "/Users/zheng/Downloads/100cdr/"
filenames = next(walk(data_path), (None, None, []))[2]
sentences = list()
for file in filenames:
    if file.endswith('.json'):
        for annotation in json.load(open(data_path+file))['annotations']:
            if annotation['label'] == 'qntfy-key-sentence-annotator':
                for content in annotation['content']:
                    sentences += [content['value']]

vectors_list = get_vectors(sentences)
vectors = list()
words = list()
for v in vectors_list:
    centers, _ = cluster_km(vectors_list[v], 4)
    for i in centers:
        words.append(v)
        vectors.append(i)

_, cluster_ids = cluster_km(vectors, 1000)
clusters = defaultdict(set)
for i, c_id in enumerate(cluster_ids):
    clusters[int(c_id)].add(words[i])
print (json.dumps(clusters))






