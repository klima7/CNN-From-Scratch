import numpy as np
import random

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from einops import rearrange


def plot_random_images(images):
    chosen_images = random.choices(images, k=16)
    joined_image = rearrange(chosen_images, '(nh nw) h w c -> (nh h) (nw w) c', nw=4)
    cmap = 'gray' if images.shape[-1] == 1 else None
    plt.imshow(joined_image, cmap=cmap)


def show_final_results(model, data):
    predictions = model.predict(data['x'])
    predicted_classes = np.argmax(predictions, axis=1)

    print(classification_report(data['y'], predicted_classes))
    ConfusionMatrixDisplay.from_predictions(
        data['y'],
        predicted_classes,
        xticks_rotation='vertical',
    )


def plot_history(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.grid()
    plt.show()

    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Categorical Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.grid()
    plt.show()


def plot_history_comparison(own_history, tf_history):
    for name in ['loss', 'val_loss', 'categorical_accuracy', 'val_categorical_accuracy']:
        plt.plot(own_history[name], label='Own')
        plt.plot(tf_history[name], label='Tensoflow')
        plt.title(f'{name} comparison')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()
