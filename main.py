from gensim.models import KeyedVectors
from gensim.similarities.annoy import AnnoyIndexer
import spacy
import numpy as np
from utilities import text_cleaner, doc_cleaner
import pandas as pd

# Taking user input and preprocessing for predictions
search_query = str(input('Search: '))
print("Searching...")
data = search_query.split()

# """## Identifying Phrases"""
initial_clean = [text_cleaner(item) for item in data]

word_vectors = KeyedVectors.load('word2vec.wordvectors', mmap='r')
annoy_index = AnnoyIndexer()
annoy_index.load('annoy_index')

# Find most similar terms/phrases
try:
    approximate_neighbors = word_vectors.most_similar(initial_clean, topn=15, indexer=annoy_index)

    # Report top 10 most similar terms/phrases, excluding exact matches
    results = pd.DataFrame(approximate_neighbors, columns=['Term', 'Similarity (0-1)'])
    results = results[results['Similarity (0-1)'] < 1.0].iloc[:10]
    results['Similarity (0-1)'] = results['Similarity (0-1)'].apply(lambda x: round(x, 2))

    print("Approximate Neighbors")
    print(results)
except:
    print("Search term not found. Please try searching for a different term or phrase.")