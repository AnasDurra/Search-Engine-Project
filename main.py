from dotenv import load_dotenv
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    # load env
    load_dotenv()

    print("Hello World")
