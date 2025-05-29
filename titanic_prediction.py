import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
titanic_data = pd.read_csv('Titanic-Dataset.csv')

# Check for missing values
print(titanic_data.isnull().sum())

# Handle missing values
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())  # Fill missing Age with median
titanic_data.drop(columns=['Cabin'], inplace=True)  # Drop Cabin due to many missing values
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])  # Fill missing Embarked with mode

# Convert categorical variables to numerical
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)

# Now you can proceed with the rest of your analysis and model building

titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))


joblib.dump(model, 'titanic_model.pkl')