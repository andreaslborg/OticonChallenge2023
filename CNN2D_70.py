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

# Save the loss history to a file
def save_losses(loss_history):
    with open('epoch_70.txt', 'w') as f:
        f.write('Epoch\tTraining Loss\tValidation Loss\n')
        for idx, (train_loss, val_loss) in enumerate(loss_history.losses):
            f.write(f'{idx + 1}\t{train_loss}\t{val_loss}\n')

# Evaluate the model's performance on the validation set
def evaluate_model(model, X_val, y_val):
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f'Val loss: {val_loss}')
    print(f'Val accuracy: {val_accuracy}')

# Save the confusion matrix to a file
def save_confusion_matrix(y_val, y_val_pred):
    y_val_pred_labels = np.argmax(y_val_pred, axis=1)
    y_val_true_labels = np.argmax(y_val, axis=1)
    conf_matrix = confusion_matrix(y_val_true_labels, y_val_pred_labels)

    with open('confusion_matrix.txt', 'w') as f:
        f.write('Confusion matrix:\n')
        for row in conf_matrix:
            f.write(' '.join(str(x) for x in row))
            f.write('\n')

# Save the test set predictions to a file
def save_test_predictions(model, test_data):
    test_data_normalized = normalize_data(test_data)
    test_data_normalized = np.expand_dims(test_data_normalized, axis=-1)

    test_predictions = model.predict(test_data_normalized)
    test_pred_labels = np.argmax(test_predictions, axis=1)

    with open('predictions_70.txt', 'w') as f:
        for pred_label in test_pred_labels:
            f.write(f'{pred_label}\n')

def main():
    TEST_SIZE = 0.3
    EPOCHS = 30

    test_data, training_data, training_labels = load_data()

    X_train, X_val, y_train, y_val = train_test_split(training_data, training_labels, test_size=TEST_SIZE, random_state=42)

    X_train_normalized, X_val_normalized, y_train_one_hot, y_val_one_hot = prepare_data(X_train, X_val, y_train, y_val)

    model = create_model()

    loss_history = train_model(model, EPOCHS, X_train_normalized, y_train_one_hot, X_val_normalized, y_val_one_hot)

    save_losses(loss_history)

    evaluate_model(model, X_val_normalized, y_val_one_hot)

    y_val_pred = model.predict(X_val_normalized)
    save_confusion_matrix(y_val_one_hot, y_val_pred)

    save_test_predictions(model, test_data)

if __name__ == '__main__':
    main()
