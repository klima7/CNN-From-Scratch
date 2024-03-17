# Nnumpy
Convolutional Neural Network written from scratch using numpy with API similar to tensorflow. Library was compared with tensorflow versions of network (`demo` directory) and achieved very close results.

## Implemented Elements

### Layers
- `InputLayer`
- `DenseLayer`
- `BiasLayer`
- `ActivationLayer (relu, leaky reLu, sigmoid, tanh, sin)`
- `DropoutLayer`
- `FlattenLayer`
- `Conv2DLayer (with bias & stride)`
- `Pool2DLayer (max, min)`
- `Padding2DLayer`
- `Crop2DLayer`
- `SoftmaxLayer`

### Losses
- `MSE`
- `CCE`

### Initializers
- `ConstantInitializer`
- `RandomNormalInitializer`
- `RandomUniformInitializer`
- `GlorotUniformInitialization`

### Metrics
- `CategoricalAccuracy`

### Callbacks
- `ModelCheckpoint`
- `EarlyStopping`

## Usage Example

### Definition
```
layers = [
    nnumpy.layers.InputLayer((28, 28, 1)),
    nnumpy.layers.Conv2DLayer(32, kernel_size=3, stride=1),
    nnumpy.layers.ActivationLayer('relu'),
    nnumpy.layers.FlattenLayer(),
    nnumpy.layers.DenseLayer(128),
    nnumpy.layers.BiasLayer(),
    nnumpy.layers.ActivationLayer('relu'),
    nnumpy.layers.DropoutLayer(0.5),
    nnumpy.layers.DenseLayer(10),
    nnumpy.layers.BiasLayer(),
    nnumpy.layers.SoftmaxLayer(),
]

model = nnumpy.network.Sequential(layers)
```

### Compilation
```
model.compile(
    loss='cce',
    metrics=['categorical_accuracy']
)
```

### Fitting
```
checkpoint_callback = nnumpy.callbacks.ModelCheckpoint('checkpoint.dat')

history = model.fit(
    train_x,
    train_y,
    validation_data=(test_x], test_y),
    learning_rate=0.001,
    epochs=10,
    callbacks=[checkpoint_callback],
)
```

### Predicting
```
predictions = model.predict(test_x)
```
