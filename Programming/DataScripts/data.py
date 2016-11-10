import Programming.TensorFlow.configuration as conf


class Data(object):
    def __init__(self, images, labels, names, ishard):
        self.data = (images, labels, names, ishard)
        self.images = images
        self.labels = labels
        self.names = names
        self.ishard = ishard

    def size(self):
        return self.images.shape[0]

    def getData(self):
        return self.data

    def applyPermutation(self, permutation):
        new_data = map(lambda x: x[permutation], self.data)
        self.__init__(*new_data)
        return self

    def createData(self, from_i, to_i):
        new_data = map(lambda x: x[from_i:to_i], self.data)
        return Data(*new_data)


class FullData(Data):
    def getOriginalDatasetSize(self):
        divide_by = 8 if conf.EXTENDED_DATASET else 1
        original_dataset_size = self.size() / divide_by
        return original_dataset_size

    def createData(self, from_i, to_i):
        new_data = map(lambda x: x[from_i:to_i], self.data)
        return FullData(*new_data)