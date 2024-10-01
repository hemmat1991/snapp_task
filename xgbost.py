import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

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

# Training the AdaBoost Model
base_estimator = DecisionTreeClassifier(max_depth=1)  # Decision stump
ada_classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100)

# Fit the model
ada_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred_ada = ada_classifier.predict(X_test)

# Training the XGBoost Model
# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost
params = {
    'objective': 'binary:logistic',
    'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Class weight
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 3,
}

# Train the model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions
y_pred_xgb = xgb_model.predict(dtest)
y_pred_xgb = (y_pred_xgb > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluation for AdaBoost
print("AdaBoost Classification Report:")
print(classification_report(y_test, y_pred_ada))

# Evaluation for XGBoost
# print("XGBoost Classification Report:")
# print(classification_report(y_test, y_pred_xgb))
#
#
# xgb_model = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight)
#
# # Set parameters for grid search
# param_grid = {
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [100, 200, 300],
#     'subsample': [0.5, 0.75, 1]
# }
#
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='f1', cv=3)
# grid_search.fit(X_train_resampled, y_train_resampled)
#
# print("Best parameters found: ", grid_search.best_params_)
# best_xgb_model = grid_search.best_estimator_
#
# # Make predictions using the best model
# y_pred_best_xgb = best_xgb_model.predict(X_test)
# print("Optimized XGBoost Classification Report:")
# print(classification_report(y_test, y_pred_best_xgb))

# AdaBoost Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.98      0.78      0.87      1468
#            1       0.03      0.28      0.05        32
#
#     accuracy                           0.77      1500
#    macro avg       0.50      0.53      0.46      1500
# weighted avg       0.96      0.77      0.85      1500


# AdaBoost Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.98      0.78      0.87      1468
#            1       0.03      0.28      0.05        32
#
#     accuracy                           0.77      1500
#    macro avg       0.50      0.53      0.46      1500
# weighted avg       0.96      0.77      0.85      1500