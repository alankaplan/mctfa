import pickle


class dat:
    def __init__(self):
        pass

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(path, 'rb') as f:
            temp_dict = pickle.load(f)
        self.__dict__.update(temp_dict)
