"""
inverted_index.py
"""

import json
import collections
import pandas as pd
from transformers import BertTokenizer


def ConstructInvertedIndex(series):
    """
    Construct an inverted index for a pandas series
    """
    
    term_frequency =  collections.Counter()
    inverted_index = dict()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # iterate through documents
    for index, text in series.iteritems():
        # catch invalid inputs
        if type(text) is not str:
            continue
        # convert text to term frequencies
        tokens = tokenizer.tokenize(text)
        terms = collections.Counter(tokens)
        term_frequency += terms

        for term in terms:
            if term in inverted_index:
                inverted_index[term].append(index)
            else:
                inverted_index[term] = [index]

    # combine frequency lookup and inverted index
    frequency_df = pd.DataFrame(term_frequency, index=[0]).T.reset_index()
    frequency_df = frequency_df.rename(columns={0: 'frequency', 'index': 'term'})
    frequency_df = frequency_df.sort_values(by='frequency', ascending=False)

    inverted_index = {str(key): str(value) for key, value in inverted_index.items()}
    index_df = pd.DataFrame(inverted_index, index=[0]).T.reset_index()
    index_df = index_df.rename(columns={0: 'inverted_index', 'index': 'term'})

    summary_df = pd.merge(frequency_df, index_df, on='term')

    return summary_df