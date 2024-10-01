import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the training dataset
train_data = pd.read_csv('task_train.csv')  # Replace with your actual training file path

# Load the test dataset
test_data = pd.read_csv('task_test.csv')  # Replace with your actual test file path

# Display the first few rows of the training dataset
print("Training Data:")
print(train_data.head())

# Display the first few rows of the test dataset
print("Test Data:")
print(test_data.head())

# Handle missing values (if necessary)
train_data = train_data.fillna('')
test_data = test_data.fillna('')

# Encode the 'Comment' column
vectorizer = CountVectorizer()

# Fit the vectorizer only on the training data and transform both datasets
X_comments_train = vectorizer.fit_transform(train_data['Comment'])
X_comments_test = vectorizer.transform(test_data['Comment'])

# Create feature DataFrame for training
X_train = pd.concat([train_data[['Income', 'Time', 'Origin', 'Destination']],
                      pd.DataFrame(X_comments_train.toarray(), columns=vectorizer.get_feature_names_out())], axis=1)

# Create feature DataFrame for testing
X_test = pd.concat([test_data[['Income', 'Time', 'Origin', 'Destination']],
                     pd.DataFrame(X_comments_test.toarray(), columns=vectorizer.get_feature_names_out())], axis=1)

# Define labels
y_train = train_data['Label']
y_test = test_data['Label']

# Apply SMOTE to the training data to handle class imbalance
print("Class distribution in training set before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution in training set after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Neural Network Model Definition
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluation for the Neural Network
print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred))