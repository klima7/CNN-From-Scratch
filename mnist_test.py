from tensorflow.keras.datasets.mnist import load_data as load_data_MNIST
(x_train, y_train), (x_test, y_test) = load_data_MNIST()

x_train = x_train.flatten()
x_test = x_test.flatten()

layers = [
    InputLayer((32*32,)),
    DenseLayer(10),
    ActivationLayer(LeakyReLuActivation()),
]

own_model = NeuralNetwork(layers, epochs=1500, learning_rate=0.01)