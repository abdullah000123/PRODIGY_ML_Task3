import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
import os
import cv2
from sklearn.utils import shuffle

train_data_set = '/content/task3/train/train'
test_data_set = '/content/task3/test1/test1'

# Initialize lists for labels and images
label = []
images = []

# Load training data
for image in os.listdir(train_data_set):
    if image.endswith('.jpg') and image[:3] == 'cat':
        label = 1
    elif image.endswith('.jpg') and image[:3] == 'dog':
        label = 0

    image_path = os.path.join(train_data_set, image)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resize_image = cv2.resize(img, (120, 120))
    resize_image = resize_image.astype(np.float32) / 255.0  # Normalize
    images.append([resize_image, label])

# Prepare features and labels
feature = []
label_out = []
for x, y in images:
    feature.append(x)
    label_out.append(y)

# Shuffle data
feature, label_out = shuffle(feature, label_out)

# Load test data
test_img = []
for img in os.listdir(test_data_set):
    if img.endswith('.jpg') and img[:3] == 'dog':
        label = 0
    else:
        label = 1

    img_path = os.path.join(test_data_set, img)
    test_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resz_test_img = cv2.resize(test_image, (120, 120))
    resz_test_img = resz_test_img.astype(np.float32) / 255.0  # Normalize
    test_img.append([resz_test_img, label])

# Prepare test features and labels
test_input_feature = []
test_out_label = []
for x, y in test_img:
    test_input_feature.append(x)
    test_out_label.append(y)

# Convert test features and labels to NumPy arrays
test_input_feature = np.array(test_input_feature)
test_out_label = np.array(test_out_label)

# Initialize SVM classifier
model = SVC()

# Batch training
batch_size = 300
print('Starting to train the model...')

for i in range(0, len(feature), batch_size):
    batch_X = feature[i:i + batch_size]
    batch_y = label_out[i:i + batch_size]

    # Convert to NumPy array and reshape the batch of images to 2D for SVM
    batch_X = np.array(batch_X).reshape(len(batch_X), -1)  # Flatten the images

    print("Training for batch =", i // batch_size + 1)
    model.fit(batch_X, batch_y)

# Predict on test data
predictions = []  # Initialize an empty list to store predictions
for i in range(0, len(test_input_feature), batch_size):
    batch_X = test_input_feature[i:i + batch_size]

    # Convert to NumPy array and reshape the batch of images to 2D for SVM
    batch_X = np.array(batch_X).reshape(len(batch_X), -1)  # Flatten the images

    print("TESTING for batch =", i // batch_size + 1)
    batch_predictions = model.predict(batch_X)  # Get predictions for the current batch
    predictions.extend(batch_predictions)  # Accumulate predictions

# Convert predictions list to a NumPy array for evaluation
predictions = np.array(predictions)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(test_out_label, predictions)
print('Accuracy of model is =', accuracy)

# Use 'predictions' for confusion matrix calculation
cm = confusion_matrix(test_out_label, predictions)
print('Confusion matrix of this module is =\n', cm)
