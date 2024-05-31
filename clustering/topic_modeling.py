import gensim
import pyLDAvis
from gensim import corpora
from gensim.models import CoherenceModel

from text_processors.antique_text_processor import AntiqueTextProcessor
from common.constants import Locations
from database.mongo_helper import MongoDBConnection
from common.file_utilities import FileUtilities
import pickle

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Assuming you have a list of documents
documents = MongoDBConnection.get_instance().get_collection("antique")
documents = [doc['doc_content'] for doc in documents.find()]

# # Load your pre-trained TF-IDF vectorizer (for other purposes if needed)
# model_path = Locations.generate_model_path("antique")
# tfidf_vectorizer = FileUtilities.load_file(model_path)

# Initialize your custom text processor
text_processor = AntiqueTextProcessor()

# Preprocess documents using your custom processor
processed_docs = [text_processor.process(doc) for doc in documents]

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# var = [[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:]]
# print(var)
# Train LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   num_topics=20,
                                   id2word=dictionary,
                                   passes=10,
                                   random_state=100,
                                   alpha='auto',
                                   per_word_topics=True)

# # Save the LDA model
# lda_model.save('lda_model.model')

# Display topics
topics = lda_model.print_topics()
for topic in topics:
    print(topic)

# Visualize the topics
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')

print("LDA visualization saved to lda_visualization.html")