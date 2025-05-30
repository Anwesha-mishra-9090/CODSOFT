# Movie Rating Prediction with Python

## Project Overview

This project aims to build a machine learning model that predicts movie ratings based on features such as genre, director, and actors. By analyzing historical movie data, the model estimates ratings given by users or critics, providing insights into factors influencing movie ratings.

## Features

- Data loading with robust encoding handling.
- Data preprocessing including handling missing values and feature engineering.
- Feature engineering includes the number of actors and average director rating.
- Data visualization to explore rating distributions and genre effects.
- Model building using a Random Forest Regressor with hyperparameter tuning using RandomizedSearchCV.
- Model evaluation using Mean Squared Error, Mean Absolute Error, and R² score.
- Visualization of actual vs. predicted ratings and feature importance.

## Dataset

The dataset should be a CSV file named `movie_dataset.csv` placed in the same directory as the script. It should include columns like:

- Name
- Year
- Duration
- Genre
- Rating
- Votes
- Director
- Actor 1
- Actor 2
- Actor 3

## How to Use

1. Ensure you have Python 3 and the required libraries installed. You can install the dependencies using:
2. Place the `movie_dataset.csv` in the project directory.
3. Run the script:
4. The script will:
- Load and preprocess the data.
- Visualize the data distribution and relationships.
- Train a Random Forest regression model.
- Evaluate and print model performance metrics.
- Display plots of actual vs predicted ratings and feature importances.

## Project Structure

- `movie_rating_prediction.py`: Main script containing data processing, modeling, evaluation, and visualization logic.
- `movie_dataset.csv`: Dataset file (not included, user-provided).
- `README.md`: This file.

## Insights

The model helps understand the influence of genre, director, and actor presence on movie ratings, demonstrating how different features contribute to predicting ratings.

## License

This project is open-source and free to use.

## Contact

For questions or suggestions, please contact [mishra.anwesha143@gmail.com ]

---

Enjoy exploring movie rating predictions with Python!
