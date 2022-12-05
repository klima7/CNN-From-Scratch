import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from .losses import MseLoss


class NeuralNetwork(BaseEstimator, ClassifierMixin):

    DEFAULT_EPOCHS = 1
    DEFAULT_LEARNING_RATE = 0.001

    def __init__(self, layers, loss=None):
        self.layers = layers
        self.loss = loss or MseLoss()

        self.epochs = None
        self.learning_rate = None
        self.training = False

        self.__connect(layers)

    def __connect(self, layers):
        for i in range(len(layers)):
            layer = layers[i]
            prev_layer = layers[i-1] if i-1 >= 0 else None
            next_layer = layers[i+1] if i+1 < len(layers) else None
            layer.connect(self, prev_layer, next_layer)

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]

    def fit(self, X, Y, **kwargs):
        self.epochs = kwargs.get('epochs', self.DEFAULT_EPOCHS)
        self.learning_rate = kwargs.get('learning_rate', self.DEFAULT_LEARNING_RATE)

        self.training = True

        for epoch_no in range(self.epochs):
            self.__learn_epoch(X, Y, epoch_no+1)

        self.training = False

    def __learn_epoch(self, X, Y, epoch_no):
        Xs, Ys = self.__shuffle(X, Y)

        for x, y in tqdm(zip(Xs, Ys), total=len(X), desc=f'Epoch {epoch_no}'):
            self.__learn_single(x, y)

    def __learn_single(self, x, y):
        prediction = self.__propagate(x)
        delta = self.loss(prediction, y)
        self.__backpropagate(delta)

    def __propagate(self, x):
        for layer in self.layers:
            x = layer.propagate_with_validation(x)
        return x

    def __backpropagate(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backpropagate_with_validation(delta)

    def predict(self, X):
        predictions = [self.__propagate(x) for x in X]
        return np.array(predictions)

    def summary(self):
        for layer in self.layers:
            print(layer)
            print(f'Shape: {tuple(layer.input_shape)} -> {tuple(layer.output_shape)}')
            print('-'*50)

    @staticmethod
    def __shuffle(X, Y):
        permutation = np.random.permutation(len(X))
        return X[permutation], Y[permutation]


class BinaryNNClassifier(NeuralNetwork):

    def __init__(self, layers, loss=None):
        super().__init__(layers, loss)
        self.encoder = OneHotEncoder()

    def predict(self, X):
        output = self.input_layer.propagate(X)
        assert output.shape[1] == 1
        output = output.flatten()
        labels = (output > 0.5).astype(np.int_)
        return labels

    def predict_proba(self, X):
        output = super().predict(X)
        output = output.flatten()
        nom = np.exp(output)
        denom = np.sum(nom)
        return nom / denom


class MulticlassNNClassifier(NeuralNetwork):

    def __init__(self, layers, loss=None):
        super().__init__(layers, loss)
        self.encoder = OneHotEncoder()

    def fit(self, X, Y, **kwargs):
        encoded_Y = self.encoder.fit_transform(Y.reshape(-1, 1)).toarray()
        print(encoded_Y.shape)
        super().fit(X, encoded_Y, **kwargs)

    def predict(self, X):
        output = super().predict(X)
        labels_no = np.argmax(output, axis=1)
        labels = np.take(self.encoder.categories_, labels_no)
        return labels
