import numpy as np
import pandas as pd
import gc
from sklearn.decomposition import PCA

# DATA PREPROCESSING
str_type = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3',
    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 
    'id_28', 'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 
    'id_38', 'DeviceType', 'DeviceInfo'
]

cols = [
    'TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 
    'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 
    'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 
    'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 
    'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 
    'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]

# Extend columns with V1, V2, ... based on correlation EDA
v = [1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40, 41, 44, 47, 48]
v += [54, 56, 59, 62, 65, 67, 68, 70, 76, 78, 80, 82, 86, 88, 89, 91]
cols += ['V' + str(x) for x in v]

# Load datasets
train_transaction = pd.read_csv('/mnt/data/train_transaction.csv', index_col='TransactionID', usecols=cols + ['isFraud'])
train_identity = pd.read_csv('/mnt/data/train_identity.csv', index_col='TransactionID')
test_transaction = pd.read_csv('/mnt/data/test_transaction.csv', index_col='TransactionID', usecols=cols)
test_identity = pd.read_csv('/mnt/data/test_identity.csv', index_col='TransactionID')

# Merge datasets
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

# Target variable
y_train = train['isFraud']
X_train = train.drop(['isFraud'], axis=1)
X_test = test.copy()

# Memory reduction and encoding
for f in X_train.columns:
    if X_train[f].dtype == 'object' or X_train[f].dtype.name == 'category':
        df_comb = pd.concat([X_train[f], X_test[f]], axis=0)
        df_comb, _ = df_comb.factorize(sort=True)
        X_train[f] = df_comb[:len(X_train)]
        X_test[f] = df_comb[len(X_train):]

# Fill missing values
X_train.fillna(-1, inplace=True)
X_test.fillna(-1, inplace=True)

# PCA Transformation
pca = PCA(n_components=32, random_state=42)  # Choose the number of components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Save to CSV
pd.DataFrame(X_train_pca, index=train.index).to_csv('X_train_pca.csv')
pd.DataFrame(X_test_pca, index=test.index).to_csv('X_test_pca.csv')

# Add a column to differentiate between train and test datasets
X_train_pca['isFraud'] = y_train.values  # Add the target column to train data
X_train_pca['dataset'] = 'train'         # Label the train dataset
X_test_pca['isFraud'] = -1               # Placeholder for test data (no target available)
X_test_pca['dataset'] = 'test'           # Label the test dataset

# Combine the two datasets
credit_card_data = pd.concat([X_train_pca, X_test_pca])

# Save to a single CSV file
credit_card_data.to_csv('credit_card.csv')
