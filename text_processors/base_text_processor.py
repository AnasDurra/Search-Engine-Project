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

import calendar
import re
import dateparser
import country_converter as coco

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


 # ---------------------------------------------------------------------------

def _normalize_months(tokens: List[str]) -> List[str]:
    normalized_tokens = []
    for token in tokens:
        if token.capitalize() in calendar.month_abbr:
            full_month_name = calendar.month_name[list(calendar.month_abbr).index(token.capitalize())]
            normalized_tokens.append(full_month_name.lower())
        else:
            normalized_tokens.append(token)
    return normalized_tokens


def _normalize_days(tokens: List[str]) -> List[str]:
    # Regular expression pattern to match day of the month
    day_pattern = r'\b(\d{1,2})(?:st|nd|rd|th)?\b(?![/\-.\d])'
    # Normalize each token in the list
    normalized_tokens = []
    for token in tokens:
        match = re.match(day_pattern, token)
        if match:
            normalized_tokens.append(str(int(match.group(1))))  # Convert to integer to remove leading zeros
        else:
            normalized_tokens.append(token)
    return normalized_tokens


def _normalize_dates(tokens: List[str]) -> List[str]:
    normalized_tokens = []
    for token in tokens:
        parsed_date = dateparser.parse(token, settings={'STRICT_PARSING': True})
        if parsed_date is not None:
            # Extract day, month, and year components from the parsed date
            day = str(parsed_date.day)
            month = calendar.month_name[parsed_date.month]
            year = str(parsed_date.year)
            # Append the extracted components as names
            normalized_tokens.extend([day, month, year])
        else:
            normalized_tokens.append(token)
    return normalized_tokens


def _normalize_country_name(tokens: List[str]) -> List[str]:
    normalized_text = []
    for token in tokens:
        standard_name = coco.convert(names=token, to='name_short', not_found=None,)
        # print(standard_name)
        if standard_name is not None:
            normalized_text.append(standard_name)
        else:
            normalized_text.append(token)
    return normalized_text




   # tokens = _normalize_months(tokens)
   #      tokens = _normalize_days(tokens)
   #      tokens = _normalize_dates(tokens)  # this is slow


   # tokens = _normalize_country_name(tokens)
