import re
import numpy as np

# Utility functions for data cleaning and modeling

def select_tokens(token):
    """
    Only select valid tokens for preprocessing. If token is empty, punctuation, or stopword, return False.
    """
    if (not token or token.is_punct or not token.text.strip() or token.is_stop or len(token.text)<3):
        return False
    return True

def text_cleaner(text):
    """
    Preprocess text with regular expressions
    """
    # Noise removal using regular expressions + lowercase text
    text = re.sub(r"pic.twitter.com\S+|http\S+", '', str(text)) # Remove urls
    text = re.sub(r"[^A-Za-z']+", " ", text)
    text = text.lower()

    return text

def doc_cleaner(doc):
    """
    Remove noise from text, then preprocess with spacy.
    """
    # Lemmatize valid tokens 
    lemmatized = [token.lemma_ for token in doc if select_tokens(token)]

    return lemmatized


def shuffle_list(lst):
    """Utility function to shuffle our list for training"""
    shuffled = list(lst)
    np.random.shuffle(shuffled)
    return shuffled