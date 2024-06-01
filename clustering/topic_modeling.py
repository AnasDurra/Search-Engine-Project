import pandas as pd
from matplotlib import pyplot as plt

from common.constants import Locations
from database.mongo_helper import MongoDBConnection
from common.file_utilities import FileUtilities

from sklearn.decomposition import LatentDirichletAllocation

documents = MongoDBConnection.get_instance().get_collection("antique")
documents = [doc['doc_content'] for doc in documents.find()]

# Load your pre-trained TF-IDF vectorizer
model_path = Locations.generate_model_path("antique")
tfidf_vectorizer = FileUtilities.load_file(model_path)

# Load your pre-trained TF-IDF matrix
model_path = Locations.generate_matrix_path("antique")
doc_term_matrix = FileUtilities.load_file(model_path)

# check out the dimensions
print(f'Rows: {doc_term_matrix.shape[0]}, Columns: {doc_term_matrix.shape[1]}')

# Learn X topics from the text documents
num_topics = 5

lda_topic_model = LatentDirichletAllocation(n_components=num_topics, random_state=12345)

# Train the LDA model and get the document topic assignments
doc_topic_matrix = lda_topic_model.fit_transform(doc_term_matrix)

col_names = [f'Topic {x}' for x in range(1, num_topics + 1)]

# Display each document's topic assignments
doc_topic_df = pd.DataFrame(doc_topic_matrix, columns=col_names)
doc_topic_df.head(n=10)

# Display the top X words for each topic
num_words = 10

for topic, words in enumerate(lda_topic_model.components_):
    word_total = words.sum()  # Get the total word weight for topic
    sorted_words = words.argsort()[::-1]  # Sort in descending order
    print(f'\nTopic {topic + 1:02d}')  # Print the topic
    for i in range(0, num_words):  # Print topic's top 10 words
        word = tfidf_vectorizer.get_feature_names_out()[sorted_words[i]]
        word_weight = words[sorted_words[i]]
        print(f'  {word} ({word_weight:.3f})')

perplexity = []
topic_nums = range(2, 11)

# Calculate the perplexity metric over multiple topic counts
for topics in topic_nums:
    lda = LatentDirichletAllocation(n_components=topics, random_state=12345)
    lda.fit(doc_topic_matrix)

    perplexity.append(lda.perplexity(doc_topic_matrix))

import seaborn as sns

# Plot perplexity by topic count
ax = sns.lineplot(x=topic_nums, y=perplexity, marker='o')

# Add labels
ax.set_title('Perplexity by Topic Count')
ax.set_xlabel('Topic Count')
ax.set_ylabel('Perplexity')

# Show plot
plt.show()
