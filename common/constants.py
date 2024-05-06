from enum import Enum


class Locations:
    MODEL_PATH: str = "model"
    MATRIX_PATH: str = "matrix"

    @staticmethod
    def generate_model_path(model_name: str) -> str:
        return f"../engines/{model_name}/{Locations.MODEL_PATH}/{model_name}_model.pkl"

    @staticmethod
    def generate_matrix_path(model_name: str) -> str:
        return f"../engines/{model_name}/{Locations.MATRIX_PATH}/{model_name}_matrix.pkl"
