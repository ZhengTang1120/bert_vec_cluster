from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

from sklearn.cluster import AffinityPropagation

def get_vectors(sentences):
    vectors = list()
    words = list()
    for sent in sentences:
        sent = sent.lower()
        encoded_input = tokenizer(text, return_tensors='pt')
        for token in tokenizer.convert_ids_to_tokens(encoded_input.input_ids.data.numpy().tolist()[0]):
            words.append(token)
        output = model(**encoded_input)
        for vec in outputs.last_hidden_state[0].data.numpy():
            vectors.append(vec)
    return vectors

def cluster(vectors):
    af = AffinityPropagation(preference=-50).fit(vectors)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    return labels
