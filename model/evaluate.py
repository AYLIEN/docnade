import numpy as np
import sklearn.metrics.pairwise as pw


def closest_docs_by_index(corpus_vectors, query_vectors, n_docs):
    docs = []
    sim = pw.cosine_similarity(corpus_vectors, query_vectors)
    order = np.argsort(sim, axis=0)[::-1]
    for i in range(len(query_vectors)):
        docs.append(order[:, i][0:n_docs])
    return np.array(docs)


def precision(label, predictions):
    if len(predictions):
        return float(
            len([x for x in predictions if label in x])
        ) / len(predictions)
    else:
        return 0.0


def evaluate(
    corpus_vectors,
    query_vectors,
    corpus_labels,
    query_labels,
    recall=[0.0002]
):
    corpus_size = len(corpus_labels)
    query_size = len(query_labels)

    results = []
    for r in recall:
        n_docs = int((corpus_size * r) + 0.5)
        if not n_docs:
            results.append(0.0)
            continue

        closest = closest_docs_by_index(corpus_vectors, query_vectors, n_docs)

        avg = 0.0
        for i in range(query_size):
            doc_labels = query_labels[i]
            doc_avg = 0.0
            for label in doc_labels:
                doc_avg += precision(label, corpus_labels[closest[i]])
            doc_avg /= len(doc_labels)
            avg += doc_avg
        avg /= query_size
        results.append(avg)
    return results
