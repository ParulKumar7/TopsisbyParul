import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

models = ["deepset/sentence_bert", "google/electra-small-discriminator", "vinai/phobert-base", "allenai/longformer-base-4096", "facebook/bart-base", "microsoft/mpnet-base", "distilbert-base-uncased", "bert-base-uncased", "roberta-base", "squeezebert/squeezebert-uncased", "sentence-transformers/paraphrase-MiniLM-L6-v2"]

metrics = ["cosine_similarity", "euclidean_distance", "manhattan_distance", "minkowski_distance", "correlation_coefficient"]

data = []

t1 = """
Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and contextually relevant.
"""

t2 = """
Machine learning is a subset of artificial intelligence that involves the development of algorithms and statistical models that enable computers to perform specific tasks without explicit programming. In the context of natural language processing, machine learning algorithms are often used to analyze and understand the structure and meaning of human language.
"""

for m in models:
    model = SentenceTransformer(m)

    cosine_sim = cosine_similarity(model.encode([t1]), model.encode([t2]))[0][0]
    euclidean_dist = np.linalg.norm(model.encode([t1]) - model.encode([t2]))
    manhattan_dist = np.abs(model.encode([t1]) - model.encode([t2])).sum()
    minkowski_dist = np.power(np.power(np.abs(model.encode([t1]) - model.encode([t2])), 3).sum(), 1/3)
    jaccard_sim = len(set(t1.split()) & set(t2.split())) / len(set(t1.split()) | set(t2.split()))
    correlation_coeff = np.corrcoef(model.encode([t1])[0], model.encode([t2])[0])[0, 1]

    param_values = [cosine_sim, euclidean_dist, manhattan_dist, minkowski_dist, correlation_coeff]

    data.append([m] + param_values)

columns = ["Model"] + metrics
df = pd.DataFrame(data, columns=columns)

df_normalized = df.copy()
for param in metrics:
    df_normalized[param] = (df[param] - df[param].min()) / (df[param].max() - df[param].min())

weights = [1] * len(metrics)

weighted_normalized_matrix = df_normalized.iloc[:, 1:] * weights

positive_ideal = weighted_normalized_matrix.max(axis=0)
negative_ideal = weighted_normalized_matrix.min(axis=0)

dist_positive_ideal = np.linalg.norm(weighted_normalized_matrix - positive_ideal, axis=1)
dist_negative_ideal = np.linalg.norm(weighted_normalized_matrix - negative_ideal, axis=1)

df_normalized["Result"] = df_normalized.apply(lambda row: np.sqrt(np.sum((row - positive_ideal) ** 2)), axis=1)

df_ranked = df_normalized.sort_values(by="Result", ascending=False).reset_index(drop=True)

df_ranked.to_csv("results.csv", index=False)

print("Results saved to 'results.csv'")
