""" 
This file is used for preprocessing text into word embedding.
"""

import numpy as np
import spacy
import unidecode


def remove_accented_chars(text):
    return unidecode.unidecode(text)

def text_tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]


def preprocess_data():
    pass






if __name__ == "__main__":
    nlp = spacy.blank("en")