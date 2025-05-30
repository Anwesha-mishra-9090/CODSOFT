import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_data(file_path):
    """Load dataset with error handling and clean column names."""
    try:
        # Try reading with UTF-16 encoding
        data = pd.read_csv(file_path, encoding='utf-16', on_bad_lines='skip')
        data.columns = data.columns.str.strip().str.lower()  # Strip whitespace and convert to lowercase
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
        raise


def preprocess_data(movie_data):
    """Preprocess the movie dataset."""
    # Inspect columns before processing
    print("Available columns before preprocessing:", movie_data.columns.tolist())

    # Check required columns exist
    required_columns = ['genre', 'rating', 'votes', 'director', 'actor 1', 'actor 2', 'actor 3']
    for col in required_columns:
        if col not in movie_data.columns:
            raise KeyError(f"Required column '{col}' not found in the data.")

    # Handle missing values
    movie_data['genre'] = movie_data['genre'].fillna('Unknown')
    movie_data['rating'] = movie_data['rating'].fillna(movie_data['rating'].mean())
    movie_data['votes'] = movie_data['votes'].fillna(0)
    movie_data.dropna(subset=['director'], inplace=True)

    # Feature engineering: number of actors present
    movie_data['num_actors'] = movie_data[['actor 1', 'actor 2', 'actor 3']].notnull().sum(axis=1)

    # Average rating of movies by director
    director_avg_rating = movie_data.groupby('director')['rating'].mean().reset_index()
    director_avg_rating.columns = ['director', 'avg_director_rating']
    movie_data = movie_data.merge(director_avg_rating, on='director', how='left')

    return movie_data


def visualize_data(movie_data):
    """Visualize the distribution of ratings and relationships."""
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_data['rating'], bins=20, kde=True)
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(12, 7))
    sns.boxplot(x='genre', y='rating', data=movie_data)
    plt.title('Movie Ratings by Genre')
    plt.xticks(rotation=45)
    plt.show()


def train_model(X, y):
    """Train the Random Forest model with hyperparameter tuning."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['num_actors', 'avg_director_rating']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['genre', 'director'])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=42)
    search.fit(X_train, y_train)

    print(f"Best parameters found: {search.best_params_}")

    return search.best_estimator_, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'R^2 Score: {r2:.4f}')

    return y_test, y_pred


def plot_results(y_test, y_pred):
    """Plot actual vs predicted ratings."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Actual vs Predicted Movie Ratings')
    plt.grid(True)
    plt.show()


def plot_feature_importances(model):
    """Plot feature importances."""
    importances = model.named_steps['regressor'].feature_importances_
    num_features = ['num_actors', 'avg_director_rating']

    # Get OneHotEncoder feature names for categorical features
    cat_transformer = model.named_steps['preprocessor'].transformers_[1][1]
    cat_features_genre = cat_transformer.get_feature_names_out(['genre'])
    cat_features_director = cat_transformer.get_feature_names_out(['director'])

    feature_names = np.concatenate([num_features, cat_features_genre, cat_features_director])

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()


def main():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'movie_dataset.csv')

    movie_data = load_data(file_path)
    movie_data = preprocess_data(movie_data)

    visualize_data(movie_data)

    # Define features and target
    X = movie_data[['genre', 'director', 'num_actors', 'avg_director_rating']]
    y = movie_data['rating']

    best_model, X_test, y_test = train_model(X, y)

    y_test, y_pred = evaluate_model(best_model, X_test, y_test)

    plot_results(y_test, y_pred)
    plot_feature_importances(best_model)


if __name__ == "__main__":
    main()
