from typing import List

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, PorterStemmer, ne_chunk
from nltk.corpus import wordnet
import string
from spellchecker import SpellChecker
import re


def get_wordnet_pos(tag_parameter):
    tag = tag_parameter[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


class BaseTextProcessor:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))
        self.spell_checker = SpellChecker(distance=4)
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = word_tokenize
        self.stemmer = PorterStemmer()
        self.pos_tagger = pos_tag

    def process(self, text) -> List[str]:
        pass

    def process_query(self, query: str) -> List[str]:
        pass

    def _word_tokenizer(self, text: str) -> List[str]:
        tokens = self.tokenizer(text)
        return tokens

    @staticmethod
    def _lowercase_tokens(tokens: List[str]) -> List[str]:
        return [str(np.char.lower(token)) for token in tokens]

    def _filter_stop_words(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]

    @staticmethod
    def _remove_registered_markers(tokens: List[str]) -> List[str]:
        return [re.sub(r'\u00AE', '', token) for token in tokens if token is not None]

    @staticmethod
    def _strip_punctuation(tokens: List[str]) -> List[str]:
        return [
            token.translate(str.maketrans('', '', string.punctuation))
            for token in tokens if token is not None
        ]

    @staticmethod
    def _eliminate_whitespaces(tokens: List[str]) -> List[str]:
        return [re.sub(r'_', ' ', token) for token in tokens if token is not None]

    @staticmethod
    def _remove_apostrophes(tokens: List[str]) -> List[str]:
        return [str(np.char.replace(token, "'", " ")) for token in tokens if token is not None]

    @staticmethod
    def _normalize_abbreviations(tokens: List[str]) -> List[str]:
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

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        lemmatizer = self.lemmatizer
        pos_tags = self.pos_tagger(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tags]
        return lemmatized_tokens

    def _spell_check(self, tokens: List[str]) -> List[str]:
        return [self.spell_checker.correction(word) if isinstance(word, str) else word for word in tokens if
                word is not None]
    @staticmethod
    def get_tokens_as_string(tokens: List[str]) -> str:
        return ' '.join(tokens)

    def _apply_named_entity_recognition(self, tokens: List[str]) -> List[str]:
        tagged_tokens = self.pos_tagger(tokens)
        ner_tags = ne_chunk(tagged_tokens)
        entities = []

        def extract_entity_text(chunk):
            if isinstance(chunk, nltk.tree.Tree):
                return ' '.join([extract_entity_text(c) for c in chunk])
            else:
                return chunk[0]

        for chunk in ner_tags:
            if hasattr(chunk, 'label') and chunk.label() == 'NE':
                entity = extract_entity_text(chunk.leaves())
                entities.append(str(entity))
            else:
                entity = chunk[0] if isinstance(chunk, nltk.tree.Tree) else chunk[0][0]
                entities.append(str(entity))

        return entities
    # def _normalize_country_name(self, tokens: List[str]) -> List[str]:
    #     normalized_text = []
    #     for token in tokens:
    #         standard_name = self.coco.convert(names=token, to='ISO3', not_found=None)
    #         if standard_name is not None:
    #             normalized_text.append(standard_name)
    #         else:
    #             normalized_text.append(token)
    #     return normalized_text

    # @staticmethod
    # def _normalize_months(tokens: List[str]) -> List[str]:
    #     normalized_tokens = []
    #     for token in tokens:
    #         if token.capitalize() in calendar.month_abbr:
    #             full_month_name = calendar.month_name[list(calendar.month_abbr).index(token.capitalize())]
    #             normalized_tokens.append(full_month_name.lower())
    #         else:
    #             normalized_tokens.append(token)
    #     return normalized_tokens

    # @staticmethod
    # def _normalize_days(tokens: List[str]) -> List[str]:
    #     # Regular expression pattern to match day of the month
    #     day_pattern = r'\b(\d{1,2})(?:st|nd|rd|th)?\b(?![/\-.\d])'
    #     # Normalize each token in the list
    #     normalized_tokens = []
    #     for token in tokens:
    #         match = re.match(day_pattern, token)
    #         if match:
    #             normalized_tokens.append(str(int(match.group(1))))  # Convert to integer to remove leading zeros
    #         else:
    #             normalized_tokens.append(token)
    #     return normalized_tokens

    # @staticmethod
    # def _normalize_dates(tokens: List[str]) -> List[str]:
    #     normalized_tokens = []
    #     for token in tokens:
    #         parsed_date = dateparser.parse(token, settings={'STRICT_PARSING': True})
    #         if parsed_date is not None:
    #             # Extract day, month, and year components from the parsed date
    #             day = str(parsed_date.day)
    #             month = calendar.month_name[parsed_date.month]
    #             year = str(parsed_date.year)
    #             # Append the extracted components as names
    #             normalized_tokens.extend([day, month, year])
    #         else:
    #             normalized_tokens.append(token)
    #     return normalized_tokens

# ---------------------------------------------------------------------------


# tokens = _normalize_months(tokens)
#      tokens = _normalize_days(tokens)
#      tokens = _normalize_dates(tokens)  # this is slow


# tokens = _normalize_country_name(tokens)
