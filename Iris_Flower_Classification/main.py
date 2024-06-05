
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the data
data = pd.read_csv("C://Users//user//Downloads//archive//Iris.csv")

# Encode the target labels
label_encoder = LabelEncoder()
data['Species'] = label_encoder.fit_transform(data['Species'])

# Split the data into features (X) and target (y)
X = data.drop(columns=['Id', 'Species'])
y = data['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
classifier = DecisionTreeClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Example new data point (sepal length, sepal width, petal length, petal width)
new_data = pd.DataFrame([[7.1,3.0,5.9,2.]], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

# Predict the species
prediction = classifier.predict(new_data)
predicted_species = label_encoder.inverse_transform(prediction)

# Print and speak the predicted species
prediction_text = f'Predicted species: {predicted_species[0]}'
print(prediction_text)
engine.say(prediction_text)
engine.runAndWait()
