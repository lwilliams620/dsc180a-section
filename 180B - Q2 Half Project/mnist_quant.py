import larq
import tensorflow as tf
tf.random.set_seed(1234)
import numpy as np
np.random.seed(1234)
import quantize

b = 2 # number of bits to use
m = 2**b - 1

print(f"Max val: {m}\nBits: {b}")

larq.quantizers.ste_sign = quantize.ste_sign

def our_tanh(x: tf.Tensor) -> tf.Tensor:   
    f2 = lambda x: (tf.math.exp(x) - tf.math.exp(-x)) / (tf.math.exp(x) + tf.math.exp(-x))
    
    f3 = lambda x: f2(4*x)
    
    f5 = lambda x: f3(x - 2*tf.math.round(x/2)) + 2*tf.math.round(x/2)
    
    return tf.clip_by_value(f5(x), -m, m)

if __name__ == "__main__":
    # BN parameters
    batch_size = 100
    alpha = 0.1
    epsilon = 1e-4

    # MLP parameters
    num_units = 4096
    n_hidden_layers = 3

    # Training parameters
    num_epochs = 1000

    # Dropout parameters
    dropout_in = 0.2
    dropout_hidden = 0.5

    # LR
    LR = 0.001

    # Patience
    patience = 5
    
    # Activation
    act = our_tanh

    # Model and weight paths
    model_path = f"mnist_{m}_{b}.h5"

    # Load Data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape data
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)

    # Turn images into grayscale
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255

    # Onehot the targets
    y_train = np.float32(np.eye(10)[y_train])
    y_test = np.float32(np.eye(10)[y_test])

    # Build MLP
    mlp = tf.keras.Sequential()
    mlp.add(tf.keras.layers.InputLayer(input_shape=(28*28)))
    mlp.add(tf.keras.layers.Dropout(rate=dropout_in))
    
    for i in range(n_hidden_layers):
        mlp.add(larq.layers.QuantDense(units=num_units, kernel_quantizer=larq.quantizers.SteSign(clip_value=[m, 2**b]), kernel_constraint=larq.constraints.WeightClip(clip_value=m)))
        mlp.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))
        mlp.add(tf.keras.layers.Activation(activation=act))
        mlp.add(tf.keras.layers.Dropout(rate=dropout_hidden))
    
    mlp.add(larq.layers.QuantDense(units=10, kernel_quantizer=larq.quantizers.SteSign(clip_value=[m, 2**b]), kernel_constraint=larq.constraints.WeightClip(clip_value=m)))
    mlp.add(tf.keras.layers.BatchNormalization(momentum=alpha, epsilon=epsilon))

    mlp.compile(loss="squared_hinge", optimizer=tf.keras.optimizers.Adam(learning_rate=LR), metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)
    
    mc = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

    mlp.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=1/6, callbacks=[es, mc])
    
    mlp.load_weights(model_path)

    results = mlp.evaluate(X_test, y_test, verbose=1)
    print('Loss: ' + str(results[0]))
    print('Accuracy: ' + str(results[1]))
    print('Error: ' + str(1-results[1]))
