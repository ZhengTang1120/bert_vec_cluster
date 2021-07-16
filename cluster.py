from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

from sklearn.cluster import KMeans

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
            if token[-1].startswith('VB') and token[0] not in ['[CLS]','[SEP]', '/'] and wnl.lemmatize(token[0], pos='v') not in ['get', 'be', 'have']:
                vectors[token[0]].append(output.last_hidden_state[0].data.numpy()[i])
            
    return vectors

def cluster(vectors):
    # af = AffinityPropagation(preference=-50).fit(vectors)
    # cluster_centers_indices = af.cluster_centers_indices_
    # labels = af.labels_
    labels = KMeans(n_clusters=1000, random_state=170).fit_predict(vectors)
    return labels


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
print (len(vectors_list))
vectors = list()
words = list()
for v in vectors_list:
    words += [v]
    vectors += [np.mean(sample(vectors_list[v], min(4, len(vectors_list[v]))), axis=0)]

cluster_ids = cluster(vectors)
clusters = defaultdict(list)
for i, c_id in enumerate(cluster_ids):
    clusters[int(c_id)].append(words[i])
print (json.dumps(clusters))






