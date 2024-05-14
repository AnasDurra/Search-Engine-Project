from dotenv import load_dotenv
from datetime import datetime
from models.tf_idf.antique_tf_idf_model import AntiqueTFIDFModel
from models.tf_idf.base_tf_idf_model import BaseTFIDFModel

print("""
_____ _____ _____

1. Antique
2. Wikipidia

_____ _____ _____
""")
# Load environment variables from .env file
load_dotenv()

action_id = int(input("Please select dataset to train (1 or 2): "))
model: BaseTFIDFModel
if action_id == 1:
    now = datetime.now()
    print("Start Time:" + now.strftime("%Y-%m-%d %H:%M:%S"))
    model = AntiqueTFIDFModel()
    model.train()
    print("Model Trained Successfully")
elif action_id == 2:
    # model = TODO: CREATE AN INSTANCE OF WIKIPIDIA MODEL
    pass
else:
    print("Invalid option")

now = datetime.now()
print("End Time:" + now.strftime("%Y-%m-%d %H:%M:%S"))
