import numpy as np
np.random.seed(1234)
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    # BN parameters
    batch_size = 100
    alpha = .1
    epsilon = 1e-4

    # MLP parameters
    num_units = 4096
    n_hidden_layers = 3

    # Training parameters
    num_epochs = 50

    # Dropout parameters
    dropout_in = .2 # 0. means no dropout
    dropout_hidden = .5

    # LR
    LR = .001

    # Model and weight paths
    model_path = "mnist_model.json"
    weights_path = "mnist_weights.h5"

    # Load Data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)

    # Turn images into grayscale
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Onehot the targets
    y_train = np.float32(np.eye(10)[y_train])
    y_test = np.float32(np.eye(10)[y_test])

    # Build MLP
    mlp = Sequential()
    mlp.add(InputLayer(input_shape=(28*28)))
    mlp.add(Dropout(rate=dropout_in))

    for k in range(n_hidden_layers):
        mlp.add(Dense(units=num_units))
        mlp.add(BatchNormalization(momentum=alpha, epsilon=epsilon))
        mlp.add(Activation('tanh'))
        mlp.add(Dropout(rate=dropout_hidden))

    mlp.add(Dense(units=10))
    mlp.add(BatchNormalization(momentum=alpha, epsilon=epsilon))

    mlp.compile(loss='squared_hinge', optimizer=Adam(learning_rate=LR), metrics=['accuracy'])

    mlp.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=1/6)

    test_results = mlp.evaluate(X_test, y_test, verbose=1)
    print('Loss: ' + str(test_results[0]))
    print('Accuracy: ' + str(test_results[1]))
    print('Error: ' + str(1-test_results[1]))
    
    with open(model_path, 'w') as f:
        f.write(mlp.to_json())

    mlp.save_weights(weights_path)
