import tensorflow as tf
tf.random.set_seed(1234)
import numpy as np
np.random.seed(1234)

if __name__ == "__main__":
    # BN parameters
    batch_size = 50
    alpha = 0.1
    epsilon = 1e-4

    # Training parameters
    num_epochs = 200

    # LR
    LR = 0.001

    # Patience
    patience = 5

    # Activation
    act = "relu"

    # Model and weight paths
    model_path = f"cifar10_FP.h5"

    # Load Data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Reshape data
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    # Turn images into grayscale
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255

    # Onehot the targets
    y_train = np.float32(np.eye(10)[y_train])
    y_test = np.float32(np.eye(10)[y_test])

    # Build CNN
    cnn = tf.keras.Sequential()

    cnn.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3)))

    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))

    cnn.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    cnn.add(tf.keras.layers.Activation(activation=act))

    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))

    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    cnn.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    cnn.add(tf.keras.layers.Activation(activation=act))

    cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same"))

    cnn.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    cnn.add(tf.keras.layers.Activation(activation=act))

    cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same"))

    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    cnn.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    cnn.add(tf.keras.layers.Activation(activation=act))

    cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same"))

    cnn.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    cnn.add(tf.keras.layers.Activation(activation=act))

    cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same"))

    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    cnn.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    cnn.add(tf.keras.layers.Activation(activation=act))
    
    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(units=1024))

    cnn.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    cnn.add(tf.keras.layers.Activation(activation=act))

    cnn.add(tf.keras.layers.Dense(units=1024))

    cnn.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    cnn.add(tf.keras.layers.Activation(activation=act))

    cnn.add(tf.keras.layers.Dense(units=10))

    cnn.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    cnn.compile(loss="squared_hinge", optimizer=tf.keras.optimizers.Adam(learning_rate=LR), metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)
    
    mc = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

    cnn.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=1/6, callbacks=[es, mc])

    cnn.load_weights(model_path)

    results = cnn.evaluate(X_test, y_test, verbose=1)
    print('Loss: ' + str(results[0]))
    print('Accuracy: ' + str(results[1]))
    print('Error: ' + str(1-results[1]))
