from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

class BaseTextProcessor:

    def process(self, text) -> List[str]:
        text_lower = self.convert_to_lowercase(text)
        processed_text = self.remove_stopwords(text_lower)
        processed_text = self.lemmatize(processed_text)  # Add lemmatization
        return processed_text

    def convert_to_lowercase(self, text: str) -> str:
        return text.lower()

    def remove_stopwords(self, text: str) -> List[str]:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_text = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        return filtered_text

    def lemmatize(self, tokens: List[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
