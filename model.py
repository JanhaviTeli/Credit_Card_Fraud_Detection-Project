import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("merchant_fraud_dataset.csv")  # Replace with your CSV filename

# Step 2: Encode categorical features
label_encoders = {}
categorical_cols = ['business_category', 'country', 'suspicious']  # Based on your image



print(df.columns)

categorical_cols = ['business_category', 'country', 'suspicious_flag']  # Adjust as per actual column names

categorical_cols = ['business_category', 'country']  # Remove 'suspicious' if not found

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    else:
        print(f"Column '{col}' not found in the dataset!")

print(df.columns)

X = df.drop(columns=['merchant_id', 'is_fraud'])  # Replace 'merchant' with the correct column name

X = df.drop(columns=['is_fraud'])  # Only drop 'is_fraud' if 'merchant' isn't there

columns_to_drop = ['merchant', 'is_fraud']
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
X = df.drop(columns=existing_columns_to_drop)
y = df['is_fraud'] if 'is_fraud' in df.columns else None

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder

# Apply LabelEncoder to all categorical columns
categorical_cols = ['merchant_id', 'suspicious_activity_flag', 'business_category', 'country']

for col in categorical_cols:
    if col in df.columns:  # Ensure the column exists in the DataFrame
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Re-split into features and target after encoding categorical columns
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Convert columns with object types to category or numeric types
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category')

# Convert columns that should be numeric (but are 'object') to float
X = X.apply(pd.to_numeric, errors='coerce')

from sklearn.preprocessing import LabelEncoder

# Initialize a LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to the categorical columns
categorical_cols = ['merchant_id', 'suspicious_activity_flag']

for col in categorical_cols:
    if col in X.columns:  # Ensure the column exists in the DataFrame
        X[col] = le.fit_transform(X[col])

# Check the column types after encoding
print(X.dtypes)

# Convert object type columns to category type
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category')

# Alternatively, convert columns to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Convert the 'merchant_id' and 'suspicious_activity_flag' to 'category' type before the train-test split
X['merchant_id'] = X['merchant_id'].astype('category')
X['suspicious_activity_flag'] = X['suspicious_activity_flag'].astype('category')

# Verify the column types
print(X.dtypes)

# Now you can perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now train the model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", enable_categorical=True)
model.fit(X_train, y_train)

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming X and y are your features and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Option 1: Encode categorical features with LabelEncoder
categorical_cols = ['merchant_id', 'suspicious_activity_flag']

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform on both training and test set for each categorical column
for col in categorical_cols:
    if col in X_train.columns:
        # Fit on the training data and transform both train and test data
        le.fit(pd.concat([X_train[col], X_test[col]], axis=0))  # Fit on both train and test
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])  # Transform test data using the same encoder

# training the model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)  # Fit on the training data

# Step 6: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")