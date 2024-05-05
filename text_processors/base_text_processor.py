from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import string


def get_wordnet_pos(tag_parameter):
    tag = tag_parameter[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


class BaseTextProcessor:

    def process(self, text) -> List[str]:
        pass

    @staticmethod
    def word_tokenizer(text: str) -> List[str]:
        return word_tokenize(text)

    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        stop_words = set(stopwords.words('english'))
        filtered_text = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        return filtered_text

    @staticmethod
    def lemmatize(tokens: List[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        pos_tags = pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tags]
        return lemmatized_tokens
