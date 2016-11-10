class DataSet(object):
    def __init__(self, train_data, validation_data, test_data):
        self.data = (train_data, validation_data, test_data)
        self.train = train_data
        self.validation = validation_data
        self.test = test_data

    def getData(self):
        return self.data
