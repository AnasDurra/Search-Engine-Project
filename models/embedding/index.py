import logging
from datetime import datetime
from dotenv import load_dotenv
from models.embedding.antique_embedding_model import AntiqueEmbeddingModel
from models.embedding.base_embedding_model import BaseEmbeddingModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("""
_____ _____ _____

1. Antique
2. Wikipedia

_____ _____ _____
""")
# Load environment variables from .env file
load_dotenv()
action_id = int(input("Please select dataset to train (1 or 2): "))
model: BaseEmbeddingModel
if action_id == 1:
    now = datetime.now()
    print("Start Time:" + now.strftime("%Y-%m-%d %H:%M:%S"))
    model = AntiqueEmbeddingModel()
    model.train()
    print("Model Trained Successfully")
elif action_id == 2:
    # model = TODO: CREATE AN INSTANCE OF WIKIPEDIA MODEL
    pass
else:
    print("Invalid option")

now = datetime.now()
print("End Time:" + now.strftime("%Y-%m-%d %H:%M:%S"))
