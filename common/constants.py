import os
from enum import Enum


class Locations:
    BASE_PATH: str = "C:/Users/anasr/Desktop/search-engine/engines"
    MODEL_PATH: str = "model"
    MATRIX_PATH: str = "matrix"
    EMBEDDINGS_PATH: str = "embeddings_model"

    # ANTIQUE COLLECTION NAME
    ANTIQUE_COLLECTION_NAME: str = "antique"
    WIKIPEDIA_COLLECTION_NAME: str = "wikipedia"

    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    ENGINES_DIR = os.path.join(ROOT_DIR, 'engines')
    CHROMA_DIR = os.path.join(ROOT_DIR, 'chroma')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    @staticmethod
    def generate_dataset_path(model_name: str) -> str:
        data_dir = os.path.join(Locations.ROOT_DIR, "data")
        return os.path.join(data_dir, model_name)

    @staticmethod
    def generate_model_path(model_name: str) -> str:
        model_dir = os.path.join(
            Locations.ENGINES_DIR,
            f"{model_name}/{Locations.MODEL_PATH}"
        )
        return os.path.join(model_dir, f"{model_name}_model.pkl")

    @staticmethod
    def generate_matrix_path(model_name: str) -> str:
        matrix_dir = os.path.join(
            Locations.ENGINES_DIR,
            f"{model_name}/{Locations.MATRIX_PATH}"
        )
        return os.path.join(matrix_dir, f"{model_name}_matrix.pkl")

    @staticmethod
    def generate_embeddings_model_path(model_name: str) -> str:
        embeddings_model_dir = os.path.join(
            Locations.ENGINES_DIR,
            f"{model_name}/{Locations.EMBEDDINGS_PATH}"
        )
        return os.path.join(embeddings_model_dir, f"{model_name}_embedding_model.pkl")

    @staticmethod
    def generate_chroma_db_path() -> str:
        if not os.path.exists(Locations.CHROMA_DIR):
            os.makedirs(Locations.CHROMA_DIR)
        return Locations.CHROMA_DIR


class DatasetNames:
    ANTIQUE: str = 'antique'
    WIKIPIDIA: str = 'wikipedia'
