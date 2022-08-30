import gensim
import os
from gensim.corpora.dictionary import Dictionary
import nltk
from nltk import WordNetLemmatizer, RegexpTokenizer
from nltk.corpus import stopwords
from scipy import stats
import numpy as np
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import math
import seaborn as sns
import sys
import random
from datetime import datetime

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

stop = set(stopwords.words('english'))

regex = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

alpha = list(np.arange(0.01, 1, 0.05))
eta = list(np.arange(0.01, 1, 0.05))


def parameter_tuning(common_corpus, common_dictionary, common_texts):
    coherence_score = []
    labels = []
    for i in range(2, 20):
        lda = gensim.models.LdaMulticore(common_corpus,
                                         id2word=common_dictionary,
                                         workers=5,
                                         num_topics=i,
                                         chunksize=100,
                                         passes=10,
                                         random_state=100)
        coherence_model_lda = CoherenceModel(model=lda,
                                             texts=common_texts,
                                             dictionary=common_dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        labels.append(i)
        coherence_score.append(coherence_lda)

    sorted_labels = [
        labels for _, labels in sorted(zip(coherence_score, labels))
    ]
    n_o_t = sorted_labels[len(sorted_labels) - 1]

    coherence_score = []
    alphas = []
    for i in alpha:
        lda = gensim.models.LdaMulticore(common_corpus,
                                         id2word=common_dictionary,
                                         alpha=i,
                                         workers=5,
                                         num_topics=n_o_t,
                                         chunksize=100,
                                         passes=10,
                                         random_state=100)
        coherence_model_lda = CoherenceModel(model=lda,
                                             texts=common_texts,
                                             dictionary=common_dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        alphas.append(i)
        coherence_score.append(coherence_lda)

    sorted_alphas = [
        alphas for _, alphas in sorted(zip(coherence_score, alphas))
    ]
    alpha_val = sorted_alphas[len(sorted_alphas) - 1]

    coherence_score = []
    etas = []
    for i in eta:
        lda = gensim.models.LdaMulticore(common_corpus,
                                         id2word=common_dictionary,
                                         alpha=alpha_val,
                                         eta=i,
                                         workers=5,
                                         num_topics=n_o_t,
                                         chunksize=100,
                                         passes=10,
                                         random_state=100)
        coherence_model_lda = CoherenceModel(model=lda,
                                             texts=common_texts,
                                             dictionary=common_dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        etas.append(i)
        coherence_score.append(coherence_lda)
    sorted_etas = [etas for _, etas in sorted(zip(coherence_score, etas))]
    eta_val = sorted_etas[len(sorted_etas) - 1]
    return n_o_t, alpha_val, eta_val


def find_rel(document_1, document_2):
    rel = 0
    for i in range(document_1.shape[1]):
        rel = rel + (document_1[0, i] * math.log2(document_2[0, i]))
    return rel


def minimum(matrix):
    rows, cols = matrix.shape
    mn = sys.maxsize
    for i in range(rows):
        for j in range(cols):
            if i != j and matrix[i, j] < mn:
                mn = matrix[i, j]
    return mn


def maximum(matrix):
    rows, cols = matrix.shape
    mx = -sys.maxsize - 1
    for i in range(rows):
        for j in range(cols):
            if i != j and matrix[i, j] > mx:
                mx = matrix[i, j]
    return mx


def normalize_r(matrix, mn, mx):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if i != j:
                matrix[i, j] = ((matrix[i, j] - mn) / (mx - mn)) - 1
    return matrix


def normalize_d(matrix, mn, mx):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if i != j:
                matrix[i, j] = ((matrix[i, j] - mn) / (mx - mn))
    return matrix


def expectation(matrix):
    rows, cols = matrix.shape
    docs = []

    for j in range(cols):
        val = 0
        for i in range(rows):
            if i != j:
                val = val + matrix[i, j]
        docs.append(val)
    return docs


def divergence(file, docs, ref_string):

    documents_list = []
    document_ids = []

    for d in range(len(docs) - 1):
        document_ids.append(d)
        string = docs[d]
        tokens = regex.tokenize(string)
        doc = []
        for j in tokens:
            if j.lower() not in stop:
                lem = lemmatizer.lemmatize(j.lower())
                doc.append(lem)
        documents_list.append(doc)

    string = ref_string
    tokens = regex.tokenize(string)
    doc = []

    for j in tokens:
        if j.lower() not in stop:
            lem = lemmatizer.lemmatize(j.lower())
            doc.append(lem)

    documents_list.append(doc)
    common_dictionary = Dictionary(documents_list)
    common_corpus = [
        common_dictionary.doc2bow(text) for text in documents_list
    ]
    n, a, e = parameter_tuning(common_corpus, common_dictionary,
                               documents_list)
    final_lda = gensim.models.LdaMulticore(common_corpus,
                                           id2word=common_dictionary,
                                           alpha=a,
                                           eta=e,
                                           workers=5,
                                           num_topics=4,
                                           chunksize=100,
                                           passes=10,
                                           random_state=100)

    lda_array = np.full((len(common_corpus), n), 0.001)
    for i in range(lda_array.shape[0]):
        vector = final_lda[common_corpus[i]]
        for j in vector:
            col = j[0]
            lda_array[i, col] = j[1]

    relevance = []
    for i in range(0, lda_array.shape[0] - 1):
        document = lda_array[i:i + 1, :]
        reference = lda_array[lda_array.shape[0] - 1:lda_array.shape[0], :]
        cur_rel = find_rel(reference, document)
        relevance.append(cur_rel)

    redundancy = 0
    ref_vector = lda_array[lda_array.shape[0] - 1:lda_array.shape[0], :]
    for i in range(ref_vector.shape[1]):
        redundancy = redundancy + (ref_vector[0, i] *
                                   math.log2(ref_vector[0, i]))

    intra_topic_r = np.zeros((lda_array.shape[0] - 1, lda_array.shape[0] - 1))
    r, c = intra_topic_r.shape
    for i in range(r):
        for j in range(c):
            if i == j:
                intra_topic_r[i, j] = np.inf
            else:
                doc_1 = lda_array[i:i + 1, :]
                doc_2 = lda_array[j:j + 1, :]
                intra_topic_r[i, j] = find_rel(doc_1, doc_2)

    redundancy_vector = []
    for i in range(0, lda_array.shape[0] - 1):
        red = 0
        d_vector = lda_array[i:i + 1, :]
        for j in range(d_vector.shape[1]):
            red = red + (d_vector[0, j] * math.log2(d_vector[0, j]))
        redundancy_vector.append(red)

    intra_topic_d = np.zeros((lda_array.shape[0] - 1, lda_array.shape[0] - 1))
    r, c = intra_topic_d.shape
    for i in range(r):
        for j in range(c):
            if i == j:
                intra_topic_d[i, j] = np.inf
            else:
                intra_topic_d[i, j] = -(intra_topic_r[i, j] -
                                        redundancy_vector[i])

    mx = maximum(intra_topic_r)
    mn = minimum(intra_topic_r)
    normalized_intra_topic_r = normalize_r(intra_topic_r, mn, mx)
    perdoc_rel = expectation(normalized_intra_topic_r)
    sns.set(font_scale=1.5)

    mx = maximum(intra_topic_d)
    mn = minimum(intra_topic_d)
    redundancy = sum(redundancy_vector) / len(redundancy_vector)
    relevance = sum(perdoc_rel) / len(perdoc_rel)

    return (redundancy, relevance)


def calculate_ids(topic_path):
    redundancy_dataset = []
    relevance_dataset = []
    doc_files = os.listdir(topic_path)
    random.shuffle(doc_files)

    print("CompletedTopics, LastTopic, Date, Redundancy, Relevance")
    for topic in doc_files:
        doc_file = open(topic_path + '/' + topic + "/documents.txt",
                        'r',
                        encoding='utf-8')
        ref_file = open(topic_path + '/' + topic + "/summary.txt",
                        'r',
                        encoding='utf-8')
        doc_string = doc_file.read()
        docs = doc_string.split("\n\n")
        ref_string = ref_file.read()
        redundancy, relevance = divergence(topic, docs, ref_string)
        redundancy_dataset.append(redundancy)
        relevance_dataset.append(relevance)
        print(str(len(redundancy_dataset)),
              str(topic),
              str(datetime.now()),
              str(sum(redundancy_dataset) / len(redundancy_dataset)),
              str(sum(relevance_dataset) / len(relevance_dataset)),
              sep=", ")


if __name__ == "__main__":
    calculate_ids("data/formatted")
