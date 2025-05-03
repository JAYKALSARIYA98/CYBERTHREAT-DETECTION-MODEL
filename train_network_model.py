import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create directories for models and visualizations
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Define the path to the dataset
dataset_path = "UNSW_NB15_training-set.csv"

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(dataset_path)

print(f"Dataset shape: {df.shape}")
print(f"Attack categories: {df['attack_cat'].unique()}")

# Count of each attack category
attack_counts = df['attack_cat'].value_counts()
print("\nAttack category distribution:")
print(attack_counts)

# Data exploration and visualization
plt.figure(figsize=(12, 6))
sns.countplot(y=df['attack_cat'], order=df['attack_cat'].value_counts().index)
plt.title('Distribution of Attack Categories')
plt.tight_layout()
plt.savefig('results/attack_distribution.png')
plt.close()

# Preprocessing
print("\nPreprocessing data...")

# Replace missing values
df.fillna(0, inplace=True)

# Separate features and target
X = df.drop(['id', 'attack_cat', 'label'], axis=1)
y_cat = df['attack_cat']  # Attack category
y_bin = df['label']       # Binary classification (0 = normal, 1 = attack)

# Identify categorical features
categorical_features = ['proto', 'service', 'state']
numeric_features = [col for col in X.columns if col not in categorical_features]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data
X_train, X_test, y_cat_train, y_cat_test, y_bin_train, y_bin_test = train_test_split(
    X, y_cat, y_bin, test_size=0.2, random_state=42)

# Create and train binary classification model (attack or not)
print("Training binary classification model...")
binary_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

binary_pipeline.fit(X_train, y_bin_train)
binary_preds = binary_pipeline.predict(X_test)

print("\nBinary Classification Results (Normal vs Attack):")
print(confusion_matrix(y_bin_test, binary_preds))
print(classification_report(y_bin_test, binary_preds))

# Create confusion matrix visualization for binary model
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_bin_test, binary_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Binary Classification')
plt.savefig('results/binary_confusion_matrix.png')
plt.close()

# Feature importance for binary model
feature_names = (
    numeric_features +
    list(binary_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
)
binary_importances = binary_pipeline.named_steps['classifier'].feature_importances_

# Sort feature importances
sorted_idx = np.argsort(binary_importances)[::-1]
top_features = sorted_idx[:15]  # Top 15 features

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_features)), binary_importances[top_features], align='center')
plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
plt.title('Top 15 Features for Attack Detection')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('results/binary_feature_importance.png')
plt.close()

# Create and train multiclass model (attack category)
print("\nTraining attack category classification model...")
multiclass_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

multiclass_pipeline.fit(X_train, y_cat_train)
multiclass_preds = multiclass_pipeline.predict(X_test)

print("\nMulticlass Classification Results (Attack Categories):")
multiclass_report = classification_report(y_cat_test, multiclass_preds, output_dict=True)
print(classification_report(y_cat_test, multiclass_preds))

# Confusion matrix for multiclass classification
plt.figure(figsize=(16, 12))
cm = confusion_matrix(y_cat_test, multiclass_preds)
unique_categories = np.unique(np.concatenate([y_cat_test, multiclass_preds]))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_categories, yticklabels=unique_categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Attack Categories')
plt.tight_layout()
plt.savefig('results/multiclass_confusion_matrix.png')
plt.close()

# Create a DataFrame with classification results for each attack category
results_df = pd.DataFrame(multiclass_report).transpose()
results_df = results_df.sort_values(by='f1-score', ascending=False)
results_df.to_csv('results/attack_classification_performance.csv')

# Plot F1 scores for each attack category
plt.figure(figsize=(12, 8))
results_df = results_df.sort_values(by='f1-score', ascending=False)
sns.barplot(x=results_df.index, y='f1-score', data=results_df)
plt.xticks(rotation=45)
plt.title('F1 Score by Attack Category')
plt.tight_layout()
plt.savefig('results/f1_score_by_category.png')
plt.close()

# Save the models
print("\nSaving models...")
joblib.dump(binary_pipeline, 'models/binary_attack_model.pkl')
joblib.dump(multiclass_pipeline, 'models/multiclass_attack_model.pkl')

# Save column information for the preprocessing pipeline
joblib.dump({
    'categorical_features': categorical_features,
    'numeric_features': numeric_features
}, 'models/feature_info.pkl')

print("Models saved successfully!")
print("Visualizations saved in the 'results' directory.") 