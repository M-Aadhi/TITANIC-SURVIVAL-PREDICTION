import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
#firstly i imoporated the needed pack for make a DecisionTree model
 
# Loading  the dataset
data = pd.read_csv('titanic.csv')

# Data preprocessing
data.dropna(subset=['Cabin'], inplace=True)
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])


#Splits the data into train and test sets
X = data.drop(['Survived'], axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define the model
model = DecisionTreeClassifier()

#Train the model
model.fit(X_train, y_train)

#prediction test
        #sample info
new_passenger_features = {
    'PassengerId':1,
    'Pclass': 3,
    'Age': 25,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 7.25,
    'Sex_female': 0,  # One-hot encoded feature for female
    'Sex_male': 1,    # One-hot encoded feature for male
    'Embarked_C': 0,  # One-hot encoded feature for Embarked C
    'Embarked_Q': 0,  # One-hot encoded feature for Embarked Q
    'Embarked_S': 1   # One-hot encoded feature for Embarked S
}
#creating a data frame with sample
new_passenger_df = pd.DataFrame([new_passenger_features])
print(new_passenger_df.columns)
# Make sample predictions
prediction = model.predict(new_passenger_df)

# the prediction
print("Prediction:", prediction)

#Evaluate the model
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
