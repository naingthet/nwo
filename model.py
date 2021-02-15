## Setup
from utilities import text_cleaner, doc_cleaner, select_tokens, shuffle_list
import numpy as np
import pandas as pd
import json
import re
import spacy
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
import multiprocessing
from gensim.models import Word2Vec
from gensim.similarities.annoy import AnnoyIndexer

# Clean data, train and save Word2Vec model, and save AnnoyIndexer model

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

## Importing Data
reddit_raw = pd.read_csv('data/reddit_raw.csv')
tweets_raw = pd.read_csv('data/tweets_raw.csv')

"""## Data Preprocessing"""

# First, load texts into lists
reddit_raw = list(reddit_raw['body'])
tweets_raw = list(tweets_raw['tweet'])
data = reddit_raw + tweets_raw

# Load spacy model
nlp = spacy.load('en_core_web_sm', exclude=["ner"])

"""## Identifying Phrases"""
initial_clean = [text_cleaner(item) for item in data]
documents = [doc_cleaner(doc) for doc in nlp.pipe(initial_clean)]
phrases = Phrases(documents, min_count=2, threshold = 0.25, scoring="npmi", connector_words=ENGLISH_CONNECTOR_WORDS)

# Phraser allows for a faster and more efficient implementation of the Phrases object
ngrams = Phraser(phrases_model=phrases)


# Add phrases to documents
docs = [ngrams[document] for document in documents]

"""## Training Word2Vec Model"""
cores = multiprocessing.cpu_count()

# Initialize the model
embedding_model = Word2Vec(
    vector_size = 300, 
    window = 10,
    min_count = 5,
    sample = 1e-4,
    workers = cores-1,
    negative = 5,
    )

# Build vocabulary table
embedding_model.build_vocab(list(docs))

# Feed shuffled documents into W2V
embedding_model.train(shuffle_list(docs), total_examples=embedding_model.corpus_count, epochs=100)

# Save the model in case we want to continue training
embedding_model.save('model.w2v')

# Save just the model's words and trained embeddings for faster loading
word_vectors = embedding_model.wv
word_vectors.save("word2vec.wordvectors")

print('Word2Vec Model Successfully Saved')

# We will use Annoy Indexer to perform word similarity queries
# Save the index to allow for faster loading

annoy_index = AnnoyIndexer(word_vectors, num_trees=200)
annoy_index.save('annoy_index')
print('AnnoyIndexer Successfully Saved')