from time import time as tm
import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# MNIST case
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# show some pictures
pictures_shown = 7
for i in range(pictures_shown):
    plt.subplot(1, pictures_shown, i + 1)
    plt.imshow(X_train[i], cmap='Greys_r')
    plt.axis('off')
plt.show()

print('label: %s' % (y_train[0:pictures_shown],))

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


def plot_accuracy_and_loss(trained_model):
    hist = trained_model.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].tick_params(colors="grey")
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy', color='grey', fontsize=16)
    ax[0].set_xlabel('Epochs', color='grey', fontsize=12)
    ax[0].set_ylabel('Accuracy', color='grey', fontsize=12)

    ax[0].legend()
    ax[0].grid(True)

    ax[1].tick_params(colors="grey")
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss', color='grey', fontsize=16)
    ax[1].set_xlabel('Epochs', color='grey', fontsize=12)
    ax[1].set_ylabel('Loss', color='grey', fontsize=12)
    ax[1].legend()
    ax[1].grid(True)

    plt.show()


# Model

model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3, 3), activation=None, padding="same", kernel_initializer='he_normal',
                  input_shape=(img_rows, img_cols, 1)))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation=None))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=None))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.4))
model2.add(Conv2D(64, kernel_size=(5, 5), strides=2, padding='same', activation=None))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, kernel_size=(5, 5), strides=2, padding='same', activation=None))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(0.4))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(num_classes, activation='softmax'))

model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# Callbacks

def scheduler(epoch, lr):
    base = 0.01
    if epoch < 6:
        return base
    elif epoch == 50 or epoch == 100:
        return base * 0.5
    elif epoch == 51 or epoch == 101:
        return base * 0.25
    else:
        return lr * 1.25 * 1 / np.round(np.sqrt(epoch), 8)


learning_rate_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="./logs")
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True)

# all callbacks
callbacks_list = [tensorboard_cb, early_stopping_cb, learning_rate_scheduler_cb]

# hyperparameters
# bss = [32, 128, 256, 512, 1024, 2048, 4096, 8192]
# bss = [256, 512, 1024]
# ess = [1, 2, 3, 5, 8, 12, 15, 30, 60, 90, 240, 350]
# ess = [1, 2, 3]

bss = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# ess = [4, 5, 8, 12, 15, 30, 60, 90, 120]
# ess = [4, 5, 8, 12, 15]

# z early stopping można 1 długie uczenie i tak się przerwie
ess = [300]

# Próba
# bss = [2048]
# ess = [2]

# model place holders
test_results = {}
best_model = {}

experiment_id = "0"  # VCS Code models <- i tak dopisuje się TYLKO do ID == 0, czyli "DEFAULT"
# experiment_id = "3"  # cb_lr_scheduler <- UWAGA tego nie widzi..
# experiment_id = mlflow.create_experiment("TF cb_lr_scheduler")
experiment = mlflow.get_experiment(experiment_id)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.keras.autolog()    # Nie działa w Windows

# hyperparams tuning
for es in ess:
    for bs in bss:

        # Not needed with EarlyStopping and fixed epochs!
        # if bs < 512 and es > 90:
        #     continue
        # elif bs < 2048 and es > 240:
        #     continue

        print("Training model with batch size = {} and {} epochs...".format(bs, es), end="", flush=True)
        tic = tm()
        history_tmp = model2.fit(X_train, y_train,
                                 batch_size=bs,
                                 epochs=es,
                                 verbose=1,
                                 validation_split=0.20,
                                 callbacks=callbacks_list
                                 )
        toc = tm()
        execution_time = round(toc - tic, 2)

        print("\b\b\b, OK! completed in {}s".format(str(execution_time)), flush=True)

        score_tmp = model2.evaluate(X_test, y_test, verbose=0)
        print("batch size == ", bs)
        print('Test loss:', score_tmp[0])
        print('Test accuracy:', score_tmp[1])
        print()

        test_results[bs] = {
            'batch_size': bs,
            'test_loss': score_tmp[0],
            'test_accuracy': score_tmp[1],
            'epochs': es,
            'execution_time': execution_time
        }

        if es >= 5:
            # plot_accuracy_and_loss(history_tmp)
            pass
        print("\n\n")

        # log to mlflow
        with mlflow.start_run(experiment_id=experiment_id) as run:

            mlflow.log_param("batch_size", bs)
            mlflow.log_param("epochs", es)
            mlflow.log_metric("test_loss", score_tmp[0])
            mlflow.log_metric("test_accuracy", score_tmp[1])
            mlflow.log_metric("execution_time", execution_time)
            mlflow.log_artifact("moje_keras_mnist_tuning callbacks.py")

            mlflow.keras.log_model(model2, f"keras_model_cb_{bs}_{es}")

        current_best = best_model.get('test_accuracy', 0.0)
        if score_tmp[1] > current_best:
            best_model = test_results[bs]

print("\nBest model summary: ")
print("-------------------\n")

for k, v in best_model.items():
    print("{}: {}".format(k, v))
