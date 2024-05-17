from typing import List
import re

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, PorterStemmer
from nltk.corpus import wordnet
import string

from spellchecker import SpellChecker


def get_wordnet_pos(tag_parameter):
    tag = tag_parameter[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


class BaseTextProcessor:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.pos_tagger = pos_tag
        self.spell_checker = SpellChecker()
        #---------------------------------
        self.stemmer = PorterStemmer()

    def process(self, text) -> List[str]:
        pass

    @staticmethod
    def _word_tokenizer(text: str) -> List[str]:
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        filtered_text = [word for word in tokens if word not in self.stop_words]
        return filtered_text

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        lemmatizer = self.lemmatizer
        pos_tags = self.pos_tagger(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tags]
        return lemmatized_tokens

    @staticmethod
    def _remove_punctuations(tokens: List[str]) -> List[str]:
        return [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens if token is not None]

    @staticmethod
    def _remove_whitespaces(tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.strip()]

    def _spell_check(self, tokens: List[str]) -> List[str]:
        return [self.spell_checker.correction(token) for token in tokens]

    # ---------------------------------------------------------------------------
    @staticmethod
    def remove_markers(tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(re.sub(r'\u00AE', '', token))
        return new_tokens

    @staticmethod
    def replace_under_score_with_space(tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(re.sub(r'_', ' ', token))
        return new_tokens

    @staticmethod
    def remove_apostrophe(tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(str(np.char.replace(token, "'", " ")))
        return new_tokens

    @staticmethod
    def normalize_abbreviations(tokens: List[str]) -> List[str]:
        resolved_terms = {}
        for token in tokens:

            if len(token) >= 2:
                synsets = wordnet.synsets(token)
                if synsets:
                    resolved_term = synsets[0].lemmas()[0].name()
                    resolved_terms[token] = resolved_term

        for abbreviation, resolved_term in resolved_terms.items():
            for i in range(len(tokens)):
                if tokens[i] == abbreviation:
                    tokens[i] = resolved_term
                    break

        return tokens

    def stemming(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(self.stemmer.stem(token))
        return new_tokens

