import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
train_data = pd.read_csv('task_train.csv')
test_data = pd.read_csv('task_test.csv')

# Separate features and target
X_train = train_data.drop(['Created_at', 'ID', 'Comment',], axis=1)
y_train = train_data['Label']

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Evaluate on test set
X_test = test_data.drop(['Created_at', 'ID', 'Comment',], axis=1)
y_test = test_data['Label']
y_pred = model.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))
