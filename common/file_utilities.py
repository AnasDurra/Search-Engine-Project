import joblib


class FileUtilities(object):
    @staticmethod
    def save_file(file_path, content):
        with open(file_path, 'wb') as f:
            joblib.dump(content, f)
            f.close()
            return file_path

    @staticmethod
    def load_file(file_path):
        with open(file_path, 'rb') as f:
            return joblib.load(f)
