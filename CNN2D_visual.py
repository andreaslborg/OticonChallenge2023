import numpy as np
import tensorflow as tf
import visualkeras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from ann_visualizer.visualize import ann_viz

# Load the dataset
test_data = np.load('test.npy')
training_data = np.load('training.npy')
training_labels = np.load('training_labels.npy')

TEST_SIZE = 0.3
EPOCHS = 1

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(training_data, training_labels, test_size=TEST_SIZE, random_state=42)

# Normalize the mel-spectrogram data
def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)

X_train_normalized = normalize_data(X_train)
X_val_normalized = normalize_data(X_val)

# Expand the dimensions of the input data to include the channel (required for CNN input)
X_train_normalized = np.expand_dims(X_train_normalized, axis=-1)
X_val_normalized = np.expand_dims(X_val_normalized, axis=-1)

# Convert labels to one-hot encoding
num_classes = 5
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

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

# Compile the model (with the Adam optimizer and categorical cross-entropy loss)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Custom callback to store losses
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append((logs.get('loss'), logs.get('val_loss')))

loss_history = LossHistory()

# Train the model
history = model.fit(X_train_normalized, y_train_one_hot, validation_data=(X_val_normalized, y_val_one_hot), epochs=EPOCHS, batch_size=32, callbacks=[loss_history])

val_loss, val_accuracy = model.evaluate(X_val_normalized, y_val_one_hot)

# Save training and validation losses to epoch.txt
with open('epoch.txt', 'w') as f:
    f.write("Epoch\tTraining Loss\tValidation Loss\n")
    for idx, (train_loss, val_loss) in enumerate(loss_history.losses):
        f.write(f"{idx + 1}\t{train_loss}\t{val_loss}\n")

# Make predictions on the validation data
y_val_pred = model.predict(X_val_normalized)
y_val_pred_labels = np.argmax(y_val_pred, axis=1)
y_val_true_labels = np.argmax(y_val_one_hot, axis=1)

# Compute and print the confusion matrix
conf_matrix = confusion_matrix(y_val_true_labels, y_val_pred_labels)
print('Confusion matrix:')
print(conf_matrix)

print(f"Val loss: {val_loss}")
print(f"Val accuracy: {val_accuracy}")

# Visualize the model
visualkeras.layered_view(
    model, 
    legend=True,
    scale_xy=2,
    scale_z=2,
    max_z=150,
    to_file='visual/arc.png'
    ).show()
