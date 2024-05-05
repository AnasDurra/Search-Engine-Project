from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import string

class BaseTextProcessor:

    def process(self, text) -> List[str]:
        text_lower = self.convert_to_lowercase(text)
        processed_text = self.remove_stopwords(text_lower)
        processed_text = self.lemmatize(processed_text)
        return processed_text

    def convert_to_lowercase(self, text: str) -> str:
        return text.lower()

    def tokenizer_word(self, text: str) -> List[str]:
        return word_tokenize(text)

    def remove_stopwords(self, text: str) -> List[str]:
        stop_words = set(stopwords.words('english'))
        tokens = self.tokenizer_word(text)
        filtered_text = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        return filtered_text

    def lemmatize(self, tokens: List[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        pos_tags = pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=self.get_wordnet_pos(tag)) for token, tag in pos_tags]
        return lemmatized_tokens

    def get_wordnet_pos(self, tag_parameter):
        tag = tag_parameter[0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
