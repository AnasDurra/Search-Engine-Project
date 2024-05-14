from enum import Enum


class Locations:
    MODEL_PATH: str = "model"
    MATRIX_PATH: str = "matrix"

    # ANTIQUE COLLECTION NAME
    ANTIQUE_COLLECTION_NAME: str = "antique"
    WIKIPEDIA_COLLECTION_NAME: str = "wikipedia"

    @staticmethod
    def generate_model_path(model_name: str) -> str:
        return f"../engines/{model_name}/{Locations.MODEL_PATH}/{model_name}_model.pkl"

    @staticmethod
    def generate_stored_model_path(model_name: str) -> str:
        return f"engines/{model_name}/{Locations.MODEL_PATH}/{model_name}_model.pkl"

    @staticmethod
    def generate_matrix_path(model_name: str) -> str:
        return f"../engines/{model_name}/{Locations.MATRIX_PATH}/{model_name}_matrix.pkl"

    @staticmethod
    def generate_stored_matrix_path(model_name: str) -> str:
        return f"engines/{model_name}/{Locations.MATRIX_PATH}/{model_name}_matrix.pkl"


class DatasetNames:
    ANTIQUE: str = 'antique'
    WIKIPIDIA: str = 'wikipedia'
