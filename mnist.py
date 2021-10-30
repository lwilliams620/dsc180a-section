import numpy as np
np.random.seed(1234)
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    # BN parameters
    batch_size = 100
    print("batch_size = "+str(batch_size))
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))

    # MLP parameters
    num_units = 4096
    print("num_units = "+str(num_units))
    n_hidden_layers = 3
    print("n_hidden_layers = "+str(n_hidden_layers))

    # Training parameters
    num_epochs = 1000
    print("num_epochs = "+str(num_epochs))

    # Dropout parameters
    dropout_in = .2 # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = .5
    print("dropout_hidden = "+str(dropout_hidden))

    # Decaying LR 
    LR_start = .003
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1/num_epochs)
    print("LR_decay = "+str(LR_decay))

    model_path = "mnist_model.json"
    print("model_path = "+str(model_path))
    
    weights_path = "mnist_weights.h5"
    print("weights_path = "+str(weights_path))

    print('Loading MNIST dataset...')

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)

    # Onehot the targets
    y_train = np.float32(np.eye(10)[y_train])
    y_test = np.float32(np.eye(10)[y_test])

    print('Building the MLP...')

    mlp = Sequential()
    mlp.add(InputLayer(input_shape=(28*28)))
    mlp.add(Dropout(rate=dropout_in))

    for k in range(n_hidden_layers):
        mlp.add(Dense(num_units))
        mlp.add(BatchNormalization(momentum=alpha, epsilon=epsilon))
        mlp.add(Dropout(rate=dropout_hidden))

    mlp.add(Dense(10))
    mlp.add(BatchNormalization(momentum=alpha, epsilon=epsilon))

    LR_schedule = ExponentialDecay(
        LR_start,
        decay_steps=num_epochs,
        decay_rate=LR_decay,
        staircase=True)

    mlp.compile(loss='squared_hinge', optimizer=Adam(learning_rate=LR_schedule), metrics=['accuracy'])
    mlp.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=1/6)

    test_results = mlp.evaluate(X_test, y_test, verbose=1)
    print('Loss: ' + str(test_results[0]))
    print('Accuracy: ' + str(test_results[1]))
    print('Error: ' + str(1-test_results[1]))
    
    with open(model_path, 'w') as f:
        f.write(mlp.to_json())

    mlp.save_weights(weights_path)