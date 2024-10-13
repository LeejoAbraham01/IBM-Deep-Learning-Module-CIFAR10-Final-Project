import logging
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Step 1: Define Hyperparameters and Configuration
def get_config():
    """
    Returns a configuration dictionary containing the hyperparameters for the model training.
    """
    config = {
        'input_shape': (32, 32, 3),
        'num_classes': 10,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.0001,
        'model_checkpoint_path': 'models/model_checkpoint.keras',
        'base_model_weights': 'imagenet',
        'include_top': False
    }

    return config


# Step 2: Build the VGG16-based Model
def build_model(config):
    """
    Builds a VGG16-based convolutional neural network with additional custom layers for CIFAR-10 classification.
    """
    logging.info("Building the model.")
    model = models.Sequential()

    base_model = VGG16(
        weights=config['base_model_weights'],
        include_top=config['include_top'],
        input_shape=config['input_shape']
    )

    # Freeze all layers in the base model initially
    base_model.trainable = False

    # Add the VGG16 base model
    model.add(base_model)

    # Add custom layers on top of VGG16
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    # Flatten the output from the convolutional layers
    model.add(layers.Flatten())

    # Fully connected layers for classification
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layer for CIFAR-10 classification (10 classes)
    model.add(layers.Dense(config['num_classes']))

    # Unfreeze the last two convolutional blocks of VGG16 for fine-tuning
    for layer in base_model.layers[-8:]:
        layer.trainable = True

    logging.info("Model built successfully.")
    return model


# Step 3: Compile the Model
def compile_model(model, config):
    """
    Compiles the Keras model with Adam optimizer and sparse categorical cross-entropy loss.
    """
    logging.info("Compiling the model.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    logging.info("Model compiled successfully.")


# Step 4: Define Callbacks for Training
def get_callbacks(config):
    """
    Returns a list of callbacks for early stopping and model checkpointing during training.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(config['model_checkpoint_path'], save_best_only=True, monitor='val_loss')
    ]
    logging.info("Callbacks created.")
    return callbacks


# Step 5: Train the Model with Data Augmentation
def train_model(model, train_images, train_labels, val_images, val_labels, config):
    """
    Trains the model using the training data with data augmentation and validates on the validation data.
    """
    logging.info("Starting training with data augmentation.")

    # Normalize the validation images
    val_images = val_images / 255.0

    # Define data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize training images
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Fit the generator to the training data
    datagen.fit(train_images)

    # Callbacks for early stopping and checkpointing
    callbacks = get_callbacks(config)

    # Train the model using the data generator
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=config['batch_size']),
        steps_per_epoch=len(train_images) // config['batch_size'],
        epochs=config['epochs'],
        validation_data=(val_images, val_labels),
        callbacks=callbacks
    )

    logging.info("Training completed.")
    return history


# Step 6: Save the Model
def save_model(model, save_path):
    """
    Saves the trained model to the specified path.
    """
    model.save(save_path)
    logging.info(f"Model saved to: {save_path}")


# Step 7: Load the Model
def load_model(load_path):
    """
    Loads a previously saved Keras model from the specified path.
    """
    model = tf.keras.models.load_model(load_path)
    logging.info(f"Model loaded from: {load_path}")

    return model


# Step 8: Evaluate the Model
def evaluate_model(model, test_images, test_labels, config):
    """
    Evaluates the trained model on the test dataset.
    """
    logging.info("Evaluating the model.")
    # Normalize the test images
    test_images = test_images / 255.0

    # Evaluate the model on the normalized test images
    test_loss, test_acc = model.evaluate(
        test_images,
        test_labels,
        batch_size=config['batch_size']
    )

    logging.info(f"Test accuracy: {test_acc}")
    logging.info(f"Test loss: {test_loss}")

    return test_loss, test_acc


# Step 9: Predict with the Model
def predict_with_model(model, new_images, config):
    """
    Predicts class labels for new images using the trained model.
    """
    logging.info("Making predictions.")
    # Normalize the new images
    new_images = new_images / 255.0

    # Make predictions on the new images
    predictions = model.predict(new_images, batch_size=1)

    # Convert the predicted logits into class labels
    predicted_classes = tf.argmax(predictions, axis=1)

    return predicted_classes, predictions


# Main script for running the model training, evaluation, and prediction
if __name__ == "__main__":
    config = get_config()

    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Split test set into validation set and test set
    val_images = test_images[:5000]
    val_labels = test_labels[:5000]
    test_images = test_images[5000:]
    test_labels = test_labels[5000:]

    # Build and compile the model
    model = build_model(config)
    compile_model(model, config)

    # Train the model with data augmentation
    history = train_model(model, train_images, train_labels, val_images, val_labels, config)

    # Save the trained model
    save_model(model, 'models/final_cifar10_model.keras')

    # Evaluate the model on the test dataset
    evaluate_model(model, test_images, test_labels, config)

    # Predict on a batch of new images
    new_images = test_images[:10]
    predicted_classes, predictions = predict_with_model(model, new_images, config)

    logging.info(f"Predicted classes: {predicted_classes.numpy()}")