"""this source file simplifies some of the boilerplate/overwhelming stuff behind the scenes so students don't
get distracted in the notebook."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2022, Michael Bommarito

# turn off warnings for this notebook
import warnings

warnings.filterwarnings('ignore')

# now make tensorflow quiet via env variable
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import after disabling warnings
import numpy
import pandas
import spacy
import transformers

# create a basic spacy pipeline with sm
nlp = spacy.load("en_core_web_trf")


def get_doc(text: str):
    """get the spacy doc object"""
    return nlp(text)


def get_tokens(text: str, remove_stopword: bool = True) -> list:
    """get all the tokens from a string of text"""
    doc = nlp(text)
    if remove_stopword:
        return [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    else:
        return [token.text for token in doc]


def get_tokens_and_lemmas(text: str) -> list:
    """get all the tokens and lemmas from a string of text"""
    doc = nlp(text)
    return [(token.text, token.lemma_) for token in doc]


def get_tokens_and_pos(text: str) -> list:
    """get all the tokens and part of speech from a string of text"""
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]


def get_tokens_and_dep(text: str) -> list:
    """get all the tokens and dependency from a string of text"""
    doc = nlp(text)
    return [(token.text, token.dep_) for token in doc]


def get_tokens_and_ner(text: str) -> list:
    """get all the tokens and named entity recognition from a string of text"""
    doc = nlp(text)
    return [(token.text, token.ent_type_) for token in doc if token.ent_type_ != '']
