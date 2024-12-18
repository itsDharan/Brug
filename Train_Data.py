import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data and labels from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the Random Forest classifier
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Predict on the test data
y_predict = model.predict(x_test)

# Evaluate the accuracy of the model
score = accuracy_score(y_predict, y_test)

# Print the accuracy of the model
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
