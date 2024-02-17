import numpy as np
from skimage import io, color, transform, feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load image data (replace 'path_to_cats' and 'path_to_dogs' with the actual paths to your dataset)
cat_images = io.ImageCollection(r"D:\prodigy_infotech\catdogssss\dogs\*.jpg")
dog_images = io.ImageCollection(r"D:\prodigy_infotech\catdogssss\cats\*.jpg")

# Resize images to a common size (e.g., 64x64 pixels)
image_size = (64, 64)
cat_images_resized = [transform.resize(img, image_size) for img in cat_images]
dog_images_resized = [transform.resize(img, image_size) for img in dog_images]

# Convert images to grayscale
cat_images_gray = [color.rgb2gray(img) for img in cat_images_resized]
dog_images_gray = [color.rgb2gray(img) for img in dog_images_resized]

# Flatten grayscale images
cat_data_gray = np.array([img.flatten() for img in cat_images_gray])
dog_data_gray = np.array([img.flatten() for img in dog_images_gray])

# Extract HOG features
cat_hog_features = np.array([feature.hog(img) for img in cat_images_gray])
dog_hog_features = np.array([feature.hog(img) for img in dog_images_gray])

# Create labels (0 for cats, 1 for dogs)
labels = np.concatenate([np.zeros(len(cat_data_gray)), np.ones(len(dog_data_gray))])

# Ensure labels are integers
labels = labels.astype(int)

# Combine cat and dog data
data_cat = np.column_stack((cat_data_gray, cat_hog_features))
data_dog = np.column_stack((dog_data_gray, dog_hog_features))

# Combine cat and dog data along rows (axis=0)
data = np.concatenate([data_cat, data_dog], axis=0)

# Extract features (X) and labels (y)
X = data[:, :-1]
y = data[:, -1]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)

# Standardize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# SVM classifier with different kernel and hyperparameters
classifier = SVC(kernel='rbf', C=10, gamma='scale', random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

# Convert labels to integers for confusion matrix
a1 = y_test
b1 = y_pred

# Confusion Matrix
print("Confusion Matrix\n", confusion_matrix(a1, b1))

# Accuracy
accuracy = accuracy_score(a1, b1)
print("Accuracy : ", accuracy)
print("Misclassification : ", 1 - accuracy)

# Precision, Recall, F1 Score
precision = precision_score(a1, b1, pos_label=1, average='binary')
recall = recall_score(a1, b1, pos_label=1, average='binary')
f1 = f1_score(a1, b1, pos_label=1, average='binary')

print("Precision Score : ", precision)
print("Recall Score: ", recall)
print("F1 score : ", f1)
