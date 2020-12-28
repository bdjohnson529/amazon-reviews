"""
inverted_index.py
"""

import json
import collections
from transformers import BertTokenizer


def ConstructInvertedIndex(series):
    """
    Construct an inverted index for a pandas series
    """
    
    term_frequency =  collections.Counter()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # iterate through documents
    for index, text in series.iteritems():
        # convert text to term frequencies
        tokens = tokenizer.tokenize(text)
        terms = collections.Counter(tokens)

        term_frequency += terms


    return term_frequency