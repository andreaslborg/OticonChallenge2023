import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the data from the .npy files
def load_data():
    test_data = np.load('test.npy')
    training_data = np.load('training.npy')
    training_labels = np.load('training_labels.npy')
    return test_data, training_data, training_labels

# Normalize the data (scales the values to be between 0 and 1)
def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)

# Prepare the data for training and validation
def prepare_data(X_train, X_val, y_train, y_val):
    # Normalize the data
    X_train_normalized = normalize_data(X_train)
    X_val_normalized = normalize_data(X_val)

    # Add an extra dimension for the color channel (grayscale)
    X_train_normalized = np.expand_dims(X_train_normalized, axis=-1)
    X_val_normalized = np.expand_dims(X_val_normalized, axis=-1)

    # Convert the labels to one-hot encoded vectors
    num_classes = 5
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    return X_train_normalized, X_val_normalized, y_train_one_hot, y_val_one_hot

# Create a CNN model
def create_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(32, 96, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Callback class to store the loss history for each epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append((logs.get('loss'), logs.get('val_loss')))

# Train the model
def train_model(model, EPOCHS, X_train, y_train, X_val, y_val):
    loss_history = LossHistory()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=32, callbacks=[loss_history])
    return loss_history

# Save the test set predictions to a file
def save_test_predictions(model, test_data):
    test_data_normalized = normalize_data(test_data)
    test_data_normalized = np.expand_dims(test_data_normalized, axis=-1)

    test_predictions = model.predict(test_data_normalized)
    test_pred_labels = np.argmax(test_predictions, axis=1)

    with open('predictions_100.txt', 'w') as f:
        for pred_label in test_pred_labels:
            f.write(f'{pred_label}\n')

# Prepare the data for training
def prepare_data_full(training_data, training_labels):
    # Normalize the data
    training_data_normalized = normalize_data(training_data)

    # Add an extra dimension for the color channel (grayscale)
    training_data_normalized = np.expand_dims(training_data_normalized, axis=-1)

    # Convert the labels to one-hot encoded vectors
    num_classes = 5
    training_labels_one_hot = tf.keras.utils.to_categorical(training_labels, num_classes=num_classes)

    return training_data_normalized, training_labels_one_hot

# Train the model on the full dataset
def train_model_full(model, EPOCHS, X_train, y_train):
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32)

def main():
    EPOCHS = 30

    test_data, training_data, training_labels = load_data()

    X_train_normalized, y_train_one_hot = prepare_data_full(training_data, training_labels)

    model = create_model()

    train_model_full(model, EPOCHS, X_train_normalized, y_train_one_hot)

    save_test_predictions(model, test_data)

if __name__ == '__main__':
    main()
