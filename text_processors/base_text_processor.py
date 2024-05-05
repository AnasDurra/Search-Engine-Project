from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string


class BaseTextProcessor:

    def process(self, text) -> List[str]:
        pass

    @staticmethod
    def remove_stopwords(text: str) -> List[str]:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_text = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        return filtered_text

    @staticmethod
    def lemmatize(tokens: List[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
