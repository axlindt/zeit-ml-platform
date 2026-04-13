### Train an XGBoost Random Forest on the training data ###

import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBRFClassifier

# 1. Load training data
features = ["objekt_name", "aboform_name", "lesedauer", "zahlung_weg_name"]
target = "churn"

train_df = pd.read_csv("/train_data.csv", sep=';')

# Identify categorical features for one-hot encoding
categorical_features = [col for col in features if train_df[col].dtype == 'object']

# Apply one-hot encoding to training data
train_df_encoded = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)

# Update features list to include new one-hot encoded columns
X_train = train_df_encoded.drop(columns=target)
y_train = train_df_encoded[target]

# 2. Train an XGBoost Random Forest
model = XGBRFClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_train)
y_prob = model.predict_proba(X_train)[:, 1]

accuracy = accuracy_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)
auc = roc_auc_score(y_train, y_prob)

print(f"Training Accuracy : {accuracy:.4f}")
print(f"Training F1 Score : {f1:.4f}")
print(f"Training AUC      : {auc:.4f}")

### Compute survival probabilities on the scoring dataset.

# 1. Load scoring data
score_df = pd.read_csv("/scoring_data.csv", sep=';')

# Apply one-hot encoding to scoring data, ensuring consistent columns with training data
score_df_encoded = pd.get_dummies(score_df, columns=categorical_features, drop_first=True)

# Align columns - this is crucial to ensure that the scoring data has the exact same columns as the training data
# This handles cases where a category might be present in train but not in score, or vice-versa.
missing_cols_in_score = set(X_train.columns) - set(score_df_encoded.columns)
for c in missing_cols_in_score:
    score_df_encoded[c] = 0

# Ensure the order of columns is the same as in X_train
X_score = score_df_encoded[X_train.columns]

# 2. Compute survival probabilities
probabilities = model.predict_proba(X_score)[:, 1]

score_df["churn_probability"] = probabilities
score_df.to_csv("scored_output.csv", index=False)

print(score_df[['id'] + features + ["churn_probability"]].head(10))
print(f"\nScored {len(score_df)} records → saved to scored_output.csv")
