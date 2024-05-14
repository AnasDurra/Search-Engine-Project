from models.antique_embedding_model import AntiqueEmbeddingModel
from models.base_embedding_model import BaseEmbeddingModel
from models.wikipedia_embedding_model import WikipediaEmbeddingModel

print("""
_____ _____ _____

1. Antique
2. Wikipidia

_____ _____ _____
""")
action_id = int(input("Please select dataset to train (1 or 2): "))
model: BaseEmbeddingModel
if action_id == 1:
    model = AntiqueEmbeddingModel()
    model.train()
    print("Model Antique Trained Successfully")
elif action_id == 2:
    # model = TODO: CREATE AN INSTANCE OF WIKIPEDIA MODEL
    print("Model Wiki Train started")
    model = WikipediaEmbeddingModel()
    model.train()
    print("Model Wiki Trained Successfully")
else:
    print("Invalid option")
