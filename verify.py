import logging
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Step 1: Load the saved model
def load_model(load_path):
    """
    Load a saved Keras model from the specified path.
    """
    model = tf.keras.models.load_model(load_path)
    logging.info(f"Model loaded from: {load_path}")
    return model


# Step 2: Load CIFAR-10 test data
def load_test_data():
    """
    Load and preprocess the CIFAR-10 test data.
    """
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize the test images
    test_images = test_images / 255.0

    return test_images, test_labels


# Step 3: Predict with the model
def predict_with_model(model, test_images, batch_size=1):
    """
    Make predictions on the test images using the loaded Keras model.
    """
    logging.info("Making predictions.")
    predictions = model.predict(test_images, batch_size=batch_size)

    # Convert the predicted logits into class labels
    predicted_classes = tf.argmax(predictions, axis=1)
    return predicted_classes.numpy()


# Step 4: Plot images with predicted class labels and true labels
def plot_images_with_predictions(images, predictions, true_labels=None):
    """
    Plot images with their predicted class labels and optionally the true labels.
    """
    plt.figure(figsize=(12, 6))

    for i in range(len(images)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        predicted_label = class_names[predictions[i]]

        # Title with predicted label and optionally the true label
        if true_labels is not None:
            true_label = class_names[true_labels[i][0]]
            title = f"Pred: {predicted_label}\nTrue: {true_label}"
        else:
            title = f"Pred: {predicted_label}"

        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Step 5: Plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels):
    """
    Plot the confusion matrix to evaluate the performance of the model on the test dataset.
    """
    logging.info("Generating confusion matrix.")
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# Step 6: Display accuracy
def display_accuracy(true_labels, predicted_labels, dataset_type):
    """
    Calculate and display the accuracy score for the given predictions and true labels.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    logging.info(f"Accuracy on {dataset_type} set: {accuracy:.4f}")
    print(f"Accuracy on {dataset_type} set: {accuracy:.4f}")


# Main function to verify the predictions
if __name__ == "__main__":
    """
    Main script to load a saved Keras model, make predictions on test data, and visually verify the results.
    """
    # Path to the saved model
    model_path = 'models/final_cifar10_model.keras'

    # Load the model
    model = load_model(model_path)

    # Load the test data
    test_images, test_labels = load_test_data()

    # Predict on the test dataset
    predicted_classes = predict_with_model(model, test_images)

    # Log the predicted classes
    logging.info(f"Predicted classes: {predicted_classes}")

    # Calculate accuracy for the test set
    display_accuracy(test_labels, predicted_classes, 'test')

    # Plot the confusion matrix
    plot_confusion_matrix(test_labels, predicted_classes)

    # Plot the first 10 test images with predictions
    plot_images_with_predictions(test_images[:10], predicted_classes[:10], true_labels=test_labels[:10])