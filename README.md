# Titanic Survival Prediction

This repository contains code and resources for predicting the survival of passengers on the Titanic using machine learning.

## Project Overview

The goal of this project is to build a machine learning model to predict whether a passenger on the Titanic survived or not based on various features such as age, gender, class, etc.

## Dataset

The dataset used in this project is the Titanic dataset provided by [Kaggle](https://www.kaggle.com/c/titanic/data). It includes the following files:
- `train.csv`: Training data with labels (survived or not)
- `test.csv`: Test data without labels

## Features

The dataset includes the following features:
- `PassengerId`: Unique ID for each passenger
- `Survived`: Survival status (0 = No, 1 = Yes)
- `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Ticket`: Ticket number
- `Fare`: Fare paid by the passenger
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook (optional, for running notebooks)

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
