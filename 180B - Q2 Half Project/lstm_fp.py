import tensorflow as tf
tf.random.set_seed(1234)
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

if __name__ == "__main__":
    # Parameters
    batch_size = 128
    embedding_output_dims = 15
    max_length = 300
    num_words = 5000
    num_epochs = 100
    patience = 5
    model_path = "lstm_FP.h5"

    # Load dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

    # Pad all sequences
    X_train_pad = pad_sequences(X_train, maxlen=max_length, value = 0.0) # 0.0 because it corresponds with <PAD>
    X_test_pad = pad_sequences(X_test, maxlen=max_length, value = 0.0) # 0.0 because it corresponds with <PAD>

    # Build LSTM
    lstm = Sequential()
    lstm.add(Embedding(num_words, embedding_output_dims, input_length=max_length))
    lstm.add(LSTM(10))
    lstm.add(Dense(units=1))
    lstm.add(tf.keras.layers.Activation("sigmoid"))

    lstm.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)
    
    mc = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    
    lstm.fit(X_train_pad, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=1/6, callbacks=[es, mc])
    
    lstm.load_weights(model_path)

    results = lstm.evaluate(X_test_pad, y_test, verbose=False)
    print('Loss: ' + str(results[0]))
    print('Accuracy: ' + str(results[1]))
    print('Error: ' + str(1-results[1]))