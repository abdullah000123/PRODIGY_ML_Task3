import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.svm import SVC
import os
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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
    resize_image = cv2.resize(img, (64,64))
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
trainf,valf,trainout,valout=train_test_split(feature,label_out,test_size=0.2,random_state=42)

# Load test data

# Prepare test features and labels
test_input_feature = valf
test_out_label = valout

# Convert test features and labels to NumPy arrays
test_input_feature = np.array(test_input_feature)
test_out_label = np.array(test_out_label)


# Initialize SVM classifier
model = SVC(kernel='rbf', C=100)
# Batch training
batch_size = 6300
print('Starting to train the model...')
train_p=[]
for i in range(0, len(feature), batch_size):
    batch_X = feature[i:i + batch_size]
    batch_y = label_out[i:i + batch_size]
    # Convert to NumPy array and reshape the batch of images to 2D for SVM
    batch_X = np.array(batch_X).reshape(len(batch_X), -1)  # Flatten the images
    print("Training for batch =", i // batch_size + 1)
    model.fit(batch_X, batch_y)


batch_size1 = 300

for i in range(0, len(feature), batch_size1):
    batch_X = feature[i:i + batch_size1]

    # Convert to NumPy array and reshape the batch of images to 2D for SVM
    batch_X = np.array(batch_X).reshape(len(batch_X), -1)  # Flatten the images
    pred=model.predict(batch_X)
    train_p.extend(pred)
    print('prediction for train batch ',i // batch_size1 + 1)

train_p=np.array(train_p)
ac=accuracy_score(label_out,train_p)
print('train accuracy ',ac)
bth=300
# Predict on test data
predictions = []  # Initialize an empty list to store predictions
for i in range(0, len(test_input_feature), bth):
    batch_X = test_input_feature[i:i + bth]

    # Convert to NumPy array and reshape the batch of images to 2D for SVM
    batch_X = np.array(batch_X).reshape(len(batch_X), -1)  # Flatten the images
    print("TESTING for batch =", i // bth + 1)
    batch_predictions = model.predict(batch_X)  # Get predictions for the current batch
    predictions.extend(batch_predictions)  # Accumulate predictions

# Convert predictions list to a NumPy array for evaluation
predictions = np.array(predictions)
# Calculate accuracy and confusion matrix
accuracy = accuracy_score(test_out_label, predictions)
print('test Accuracy of model is =', accuracy)

# Use 'predictions' for confusion matrix calculation
cm = confusion_matrix(test_out_label, predictions)
print('Confusion matrix of this module is =\n', cm)

import joblib

# Define the path where you want to save the model
model_path = '/content/drive/My Drive/svm_model.pkl'

# Save the model
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")

