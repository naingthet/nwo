# NWO.AI Coding Challenge

## Challenge Overview
Given a search query, find similar terms (trends) based on data from Twitter and Reddit posts. This implementation leverages Word2Vec word embeddings to return an ordered list of terms and phrases that are related to the search query.

## Challenge Approach
In the limited time provided for this challenge, the goal was to develop an accurate semantic search program that provides fast results. As such, Word2Vec was chosen as the approach to this challenge. Through the Gensim implementation, this program allows for online training/updating of the intial word embeddings, should new data become available.

## Scripts
1. `query.py`: Using the provided json key, this script will pull data from the BigQuery database, then save the data in csv files. The SQL queries in this script will prioritize recent tweets and reddit posts, as well as select for texts that contain more than 10 characters.
2. `utilities.py`: This script contains a few utility functions that will be useful in preprocessing data and training Word2Vec models
3. `model.py`: This script will load and preprocess the data saved using `query.py`. The data will then be used to construct a Word2Vec word embeddings model and a corresponding Annoy Indexer model (Annoy Indexer is an indexer that approximates cosine similarity, providing much faster results with minimal accuracy loss). The model and indexer are then saved by the script to allow for online training as necessary. This maximizes the speed of `main.py`, the primary script responsible for running the program. 
4. `main.py`: This script will accept and preprocess user input, then return the top search results. As mentioned, this script makes use of saved word vectors and indices, maximizing computation speed. 

## Potential Improvements
There are no doubt improvements that can be made to the current implementation. The following approaches were considered because they 1) involve significantly longer training times and/or 2) require more time for development and validation than currently provided for.
1. **Topic Modeling**: Construct a topic model for the training data using LDA (Latent Dirichlet Allocation) and summarize each topic in a few words. Then, accept user input and search for the topics that are most related to the user input. 
2. **Topic Clustering**: Cluster similar documents using Doc2Vec (document/paragraph-level Word2Vec) and use K Means Clustering to match user inputs to the most relevant topic clusters.