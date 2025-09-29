import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the paths to your dataset directories.
TRAIN_DIR = "C:\\Users\\1040G7\\Documents\\INTERNSHIP\\NITDA\\Data_Science_Advance\\Brain_tumor_project\\Braintumour\\train"
TEST_DIR = "C:\\Users\\1040G7\\Documents\\INTERNSHIP\\NITDA\\Data_Science_Advance\\Brain_tumor_project\\Braintumour\\validation"
TEST_DIR_FINAL = "C:\\Users\\1040G7\\Documents\\INTERNSHIP\\NITDA\\Data_Science_Advance\\Brain_tumor_project\\Braintumour\\test" # Final test directory
# Define the image size and batch size.
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64

# Create a list of the class names by looking at the directory names.
try:
    class_names = sorted(os.listdir(TRAIN_DIR))
    # Filter out any non-directory files that might exist.
    class_names = [name for name in class_names if os.path.isdir(os.path.join(TRAIN_DIR, name))]
except FileNotFoundError:
    print("Dataset directories not found. Please ensure 'brain_tumor_dataset/Training' and 'brain_tumor_dataset/Testing' exist.")
    exit()

# Set up data augmentation for the training dataset.
# Data augmentation helps to prevent overfitting by creating new, slightly modified images from the existing ones.
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Rescale pixel values to a 0-1 range.
    rotation_range=30,              # Randomly rotate images by up to 30 degrees.
    width_shift_range=0.2,          # Randomly shift images horizontally.
    height_shift_range=0.2,         # Randomly shift images vertically.
    shear_range=0.2,                # Apply shearing transformations.
    zoom_range=0.2,                 # Randomly zoom into images.
    horizontal_flip=True,           # Randomly flip images horizontally.
    fill_mode='nearest'             # Fill in new pixels created by transformations.
)

# Set up data generator for the validation/testing dataset.
# We only need to rescale the test data, no augmentation.
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data from the directory.
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load validation data from the directory.
validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Get the number of classes from the generator.
num_classes = len(train_generator.class_indices)
print(f"Number of classes detected: {num_classes}")
print(f"Class names: {class_names}")

# Load the VGG16 model with pre-trained ImageNet weights.
# We set `include_top=False` to remove the final classification layers, as we will add our own.
base_model = VGG16(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), # Define the input shape.
    include_top=False,                      # Exclude the final classification layers.
    weights='imagenet'                      # Use pre-trained weights from ImageNet.
)

# Freeze the layers of the base model so they are not trained.
# This keeps the powerful features learned by VGG16 intact.
for layer in base_model.layers:
    layer.trainable = False

# Build the new classification head for our model.
# This part will be trained on our specific brain tumor dataset.
x = base_model.output
x = Flatten()(x)                             # Flatten the output of the VGG16 model.
x = Dense(512, activation='relu')(x)         # Add a dense layer with 512 neurons and a ReLU activation.
x = Dropout(0.5)(x)                          # Add dropout to further prevent overfitting.
predictions = Dense(num_classes, activation='softmax')(x) # Add the final output layer with a neuron for each class.

# Combine the VGG16 base and our new head to create the final model.
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model.
model.compile(
    optimizer=Adam(learning_rate=0.0001),   # Use the Adam optimizer with a small learning rate.
    loss='categorical_crossentropy',        # Use categorical crossentropy for multi-class classification.
    metrics=['accuracy']                    # Track accuracy during training.
)

# Display a summary of the model architecture.
model.summary()

# Train the model.
# We use fit_generator for compatibility with the ImageDataGenerator.
history = model.fit(
    train_generator,
    epochs=15,                              # Train for 10 epochs (you can increase this for better results).
    validation_data=validation_generator    # Use the validation data to monitor performance on unseen data.
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR_FINAL,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")


# Save the trained model to a file.
model.save('my_model.h5')
print("Model trained and saved as my_model.h5")

# Plot training and validation accuracy.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training and validation loss.
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Display the plots.
plt.show()
