import os
from nltk import RegexpTokenizer
from nltk.util import ngrams

topic_path = 'data/formatted'
ref_path = topic_path

regex = RegexpTokenizer(r'\w+')


def abstract(topic, summary):

    sentences = []
    for i in topic:
        doc_string = i
        doc_sentences = doc_string.split('.')
        for d in range(len(doc_sentences)):
            doc_sentences[d] = doc_sentences[d].lower()
        sentences.extend(doc_sentences)

    doc_ngrams = []

    for i in sentences:
        tokens = regex.tokenize(i)
        ngram = list(ngrams(tokens, 1))
        doc_ngrams.extend(ngram)

    summary = summary[0:len(summary)]
    reference_sentences = summary.split(".")
    reference_ngrams = []

    for i in reference_sentences:
        tokens = regex.tokenize(i)
        ngram = list(ngrams(tokens, 1))
        reference_ngrams.extend(ngram)

    count = 0
    for i in reference_ngrams:
        if i in doc_ngrams:
            count += 1

    abstract_val.append(count / len(reference_ngrams))


doc_files = os.listdir(topic_path)
abstract_val = []
for f in doc_files:
    doc_file = open(topic_path + '/' + f + "/documents.txt",
                    'r',
                    encoding='utf-8')
    ref_file = open(ref_path + '/' + f + "/summary.txt", 'r', encoding='utf-8')
    doc_string = doc_file.read()
    docs = doc_string.split("\n\n")
    ref_string = ref_file.read()
    abstract(docs, ref_string)

print("Abstractness is: " + str(sum(abstract_val) / len(abstract_val)))
