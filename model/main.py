import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Load the dataset
credit_card_data = pd.read_csv('../creditcard.csv')

# Extract features (V1 to V28) and target (Class)
# Select columns V1 to V28 dynamically
feature_columns = [f'V{i}' for i in range(1, 29)]  # Generate V1 to V28 column names
X = credit_card_data[['Time', 'Amount'] + feature_columns]  # Add Time and Amount along with V1 to V28
Y = credit_card_data['Class']  # Target variable is Class

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# 1. **Scale the Numerical Features (Time, Amount, V1 to V28)**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. **Optional: Apply PCA to Reduce Dimensionality**
pca = PCA(n_components=10)  # You can change n_components as needed
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 3. **Train the Models**

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_pca, Y_train)
log_train_pred = log_model.predict(X_train_pca)
log_train_accuracy = accuracy_score(log_train_pred, Y_train)
log_test_pred = log_model.predict(X_test_pca)
log_test_accuracy = accuracy_score(log_test_pred, Y_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=2)
rf_model.fit(X_train_pca, Y_train)
rf_train_pred = rf_model.predict(X_train_pca)
rf_train_accuracy = accuracy_score(rf_train_pred, Y_train)
rf_test_pred = rf_model.predict(X_test_pca)
rf_test_accuracy = accuracy_score(rf_test_pred, Y_test)

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=2)
gb_model.fit(X_train_pca, Y_train)
gb_train_pred = gb_model.predict(X_train_pca)
gb_train_accuracy = accuracy_score(gb_train_pred, Y_train)
gb_test_pred = gb_model.predict(X_test_pca)
gb_test_accuracy = accuracy_score(gb_test_pred, Y_test)

# Print the accuracy scores
print('Logistic Regression - Accuracy on Training data:', log_train_accuracy)
print('Logistic Regression - Accuracy on Test data:', log_test_accuracy)

print('Random Forest - Accuracy on Training data:', rf_train_accuracy)
print('Random Forest - Accuracy on Test data:', rf_test_accuracy)

print('Gradient Boosting - Accuracy on Training data:', gb_train_accuracy)
print('Gradient Boosting - Accuracy on Test data:', gb_test_accuracy)

# Save the models and transformations
joblib.dump(log_model, 'logistic_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
joblib.dump(pca, 'pca_transform.pkl')
joblib.dump(scaler, 'scaler.pkl')

