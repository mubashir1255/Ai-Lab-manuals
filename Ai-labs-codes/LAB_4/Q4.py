# Step 1: Import Libraries
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Load the Digits Dataset
digits = datasets.load_digits()

# Step 3: Features (X) and Target (y)
X = digits.data
y = digits.target

# Step 4: Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the SVM Model (RBF kernel)
svm_model = SVC(kernel='rbf', gamma=0.001, C=10)
svm_model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = svm_model.predict(X_test)

# Step 7: Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Step 8: Show Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Visualize Misclassified Samples
misclassified_indexes = (y_test != y_pred)

# Select first 5 misclassified images for display
misclassified_images = X_test[misclassified_indexes][:5]
misclassified_true = y_test[misclassified_indexes][:5]
misclassified_pred = y_pred[misclassified_indexes][:5]

# Plot misclassified samples
plt.figure(figsize=(10, 4))
for index, (image, true_label, pred_label) in enumerate(zip(misclassified_images, misclassified_true, misclassified_pred)):
    plt.subplot(1, 5, index + 1)
    plt.imshow(image.reshape(8, 8), cmap='gray')
    plt.title(f"T:{true_label}, P:{pred_label}")
    plt.axis('off')

plt.suptitle("Misclassified Digits (True vs Predicted)")
plt.show()
