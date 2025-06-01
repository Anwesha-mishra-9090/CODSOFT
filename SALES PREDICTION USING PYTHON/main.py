import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Load the dataset
    data_path = 'sales_data.csv'  # Adjust this path to your actual data file
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File {data_path} not found. Please ensure the dataset is in the correct path.")
        return

    print("Dataset preview:")
    print(data.head())
    print("\nDataset Info:")
    data.info()

    # Check for missing values
    if data.isnull().sum().sum() > 0:
        print("\nMissing values found - applying forward fill.")
        data.fillna(method='ffill', inplace=True)

    # 1. EDA - Pairplot to visualize relationships and distributions
    # To avoid blank/duplicate figures:
    sns.pairplot(data, kind="reg", diag_kind="kde", markers="+",
                 plot_kws={'line_kws':{'color':'red'}})
    plt.suptitle('Pairplot with Regression Lines and KDE Diagonal', y=1.02)
    plt.show()
    plt.close()  # Close to prevent blank figure issues

    # Prepare features and target variable
    X = data.drop(columns='Sales', errors='ignore')
    y = data['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create a pipeline with scaling and Random Forest Regressor
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)
    print("\nModel training completed.")

    # Predictions & Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test Mean Squared Error: {mse:.4f}")
    print(f"Test R^2 Score: {r2:.4f}")

    # 2. Residual Plot to check errors vs predicted values
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot for Model Predictions')
    plt.show()
    plt.close()

    # Bonus: Actual vs Predicted scatter plot for clarity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.show()
    plt.close()

    # Cross-validation to assess model performance
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"\nCross-validated R^2 scores: {cv_scores}")
        print(f"Mean R^2 from cross-validation: {np.mean(cv_scores):.4f}")
    except Exception as e:
        print(f"Error during cross-validation: {e}")

if __name__ == '__main__':
    main()
