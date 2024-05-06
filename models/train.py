from models.antique_embedding_model import AntiqueEmbeddingModel
from models.base_embedding_model import BaseEmbeddingModel

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
    print("Model Trained Successfully")
elif action_id == 2:
    # model = TODO: CREATE AN INSTANCE OF WIKIPIDIA MODEL
    pass
else:
    print("Invalid option")
