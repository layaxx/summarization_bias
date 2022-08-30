import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

path_doc = './data/formatted'


def cos_sim(topic):
    topic_path = path_doc + '/' + topic + "/documents"
    documents = os.listdir(topic_path)
    reference_path = path_doc + '/' + topic + '/' + 'summary.txt'
    ref_matrix = np.loadtxt(reference_path, dtype=float)
    for d in documents:
        document_path = topic_path + '/' + d
        doc_matrix = np.loadtxt(document_path, dtype=float)
        if doc_matrix.ndim < 2 or ref_matrix.ndim < 2:
            continue
        sim_val = cosine_similarity(doc_matrix, ref_matrix)
        doc_importance = np.amax(sim_val, axis=1)
        doc_importance = doc_importance.flatten()
        sentences = doc_importance.shape[0]
        if sentences < 3:
            continue
        sent_per_part = int(sentences / 3)
        remaining = sentences % sent_per_part
        rows = (sentences - remaining) / sent_per_part
        temporary_importance = doc_importance[0:int(doc_importance.shape[0] -
                                                    remaining)].reshape(
                                                        int(rows),
                                                        sent_per_part)
        averages = np.mean(temporary_importance, axis=1)
        if (remaining > 0):
            averages = list(averages)
            averages.append(
                np.mean(
                    doc_importance[int(doc_importance.shape[0] -
                                       remaining):doc_importance.shape[0]]))
            averages = np.asarray(averages)
        if len(averages) > 3:
            sum_ = sum(averages[2:len(averages)])
            averages[2] = sum_ / (len(averages) - 2)
        averages = averages[0:3]
        first_prt.append(averages[0])
        second_prt.append(averages[1])
        third_prt.append(averages[2])


topics = os.listdir(path_doc)

first_prt = []
second_prt = []
third_prt = []
for t in topics:
    print(t)
    cos_sim(t)
    break

print(sum(first_prt) / len(first_prt))
print(sum(second_prt) / len(second_prt))
print(sum(third_prt) / len(third_prt))
