import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(file_path):
    """Load dataset with error handling and clean column names."""
    encodings = ['utf-16', 'utf-8', 'latin1']  # Tried encodings to handle decode errors
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            print(f"File loaded successfully with encoding: {encoding}")
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise
        except pd.errors.ParserError as e:
            print(f"Parser error with encoding {encoding}: {e}")
            continue
        except UnicodeDecodeError as e:
            print(f"Unicode decode error with encoding {encoding}: {e}")
            continue
    raise ValueError("Failed to read the file with available encodings.")

def preprocess_data(data):
    """Preprocess the dataset."""
    # Inspect the first few rows of the dataset
    print("First few rows of the dataset:")
    print(data.head())

    # Check for missing values
    print("Missing values in each column:")
    print(data.isnull().sum())

    # Convert species to categorical
    data['species'] = data['species'].astype('category')
    return data

def visualize_data(data):
    """Visualize the dataset."""
    sns.pairplot(data, hue='species', markers=["o", "s", "D"])
    plt.title('Iris Flower Dataset Pairplot')
    plt.show()

def train_model(X, y):
    """Train the Random Forest Classifier with hyperparameter tuning."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=42)
    search.fit(X, y)

    print(f"Best parameters found: {search.best_params_}")
    return search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

def plot_feature_importances(model, feature_names):
    """Plot feature importances."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

def main():
    # Load the Iris dataset
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'iris.csv')  # Ensure you have the iris.csv file in the same directory

    iris_data = load_data(file_path)
    iris_data = preprocess_data(iris_data)

    visualize_data(iris_data)

    # Define features and target
    X = iris_data.drop('species', axis=1)
    y = iris_data['species']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    best_model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)

    # Plot feature importances
    plot_feature_importances(best_model, X.columns)

if __name__ == "__main__":
    main()
