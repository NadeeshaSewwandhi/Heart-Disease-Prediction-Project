# Heart-Disease-Prediction-Project
This project is a heart disease prediction system developed using machine learning

import numpy as np : 
This command imports the NumPy library and aliases it as np. NumPy is a powerful library for numerical computing in Python, providing support for arrays, matrices, 
and a large collection of mathematical functions to operate on these data structures.

import pandas as pd : 
This command imports the pandas library and aliases it as pd. Pandas is an essential data manipulation and analysis library in Python. It provides data structures  like DataFrame and Series, which are used for handling and analyzing structured data (such as the heart disease dataset).

from sklearn.model_selection import train_test_split :
This command imports the train_test_split function from the sklearn.model_selection module. The train_test_split function is u to split a dataset into training and testing subsets. This is crucial for evaluating the performance of a machine learning model by training it on one subset and testing it on another.

from sklearn.linear_model import LogisticRegression : 
This command imports the LogisticRegression class from the sklearn.linear_model module. Logistic Regression is a statistical method used for binary classification problems. In this context, it will be used to predict the presence or absence of heart disease based on the features in the dataset.

from sklearn.metrics import accuracy_score : 
This command imports the accuracy_score function from the sklearn.metrics module. accuracy_score is a metric used to evaluate the performance  of a classification model. It calculates the ratio of correctly predicted instances to the total instances in the dataset, providing an  accuracy measure of the model's predictions.


                                             

data collection and processing



heart_dataset = pd.read_csv('/content/heart_disease_data.csv') : 
This command uses pandas to read a CSV file containing the heart disease dataset. The data is loaded into a DataFrame called heart_dataset.

heart_dataset.head() :
This command displays the first five rows of the DataFrame by default. It's useful for getting a quick overview of the dataset's structure and the first five records.

heart_dataset.tail() : 
This command displays the last five rows of the DataFrame by default. It helps you check the end of the dataset for any trailing data issues or to understand the structure of the last few records.

heart_dataset.shape : 
This command returns a tuple representing the dimensions of the DataFrame, i.e., the number of rows and columns. It's useful for understanding the size of the dataset.


heart_dataset.isnull().sum() :
This command checks for missing values in the dataset. It returns the count of missing values for each column. It's crucial for identifying columns that may need cleaning or imputation.


heart_dataset.describe() : 
This command generates descriptive statistics for the numerical columns in the DataFrame. It includes measures such as count, mean, standard deviation, minimum, 25th percentile (Q1), median (Q2), 75th percentile (Q3), and maximum. It provides a quick summary of the dataset's distribution and central tendencies.



                           

Splitting the Features and Target          






In any supervised machine learning problem, the dataset comprises features (input variables) and a target (output variable).X represents the features (all columns except the 'target' column).Y represents the target (the 'target' column, which indicates the presence or absence of heart disease).Separating features and target is crucial for training the model, as the model learns to predict the target based on the features.





Splitting the Dataset into Training and Testing Sets

To evaluate the performance and generalizability of a machine learning model, it is essential to divide the dataset into two parts: a training set and a testing set. This process is performed using the train_test_split function from sklearn.model_selection.Splitting the dataset allows the model to be trained on one subset and tested on another. This is critical for assessing the model's ability to generalize to new, unseen data. Without this split, the model might perform well on the training data but poorly on new data, leading to overfitting.

X_train and Y_train:     These subsets are used to train the model. They represent 80% of the original data.
X_test and Y_test:       These subsets are used to evaluate the model's performance. They represent the remaining 20% of the data.
test_size=0.2:           This parameter specifies that 20% of the data should be allocated to the testing set.
stratify=Y:              This parameter ensures that the split maintains the same proportion of classes in both the training and testing sets. This is particularly important for 
                         imbalanced datasets.
random_state=2:          This parameter ensures reproducibility by setting a seed for the random number generator.










Training a Logistic Regression Model

The primary goal of training the model is to allow it to learn from the historical data (training set). By fitting the model to the training data, it identifies patterns and relationships between the input features and the target variable. These learned patterns are then used to make predictions on new, unseen data.Once the model has been trained, it can be used to predict the target variable for new data points.Training the model is a necessary step before evaluating its performance. After the model is trained, it can be tested on the test data (X_test and Y_test) to see how well it generalizes to new, unseen data. This helps in assessing the model’s accuracy and its ability to make reliable predictions.

model = LogisticRegression() : This command initializes a Logistic Regression model. LogisticRegression() is a class from the sklearn.linear_model module.
At this stage, no learning has occurred yet. We have simply created an instance of the Logistic Regression model.

model.fit(X_train, Y_train) : The fit method is used to train the model on the training data (X_train and Y_train).X_train contains the features, and Y_train contains the target variable (presence or absence of heart disease).During this process, the Logistic Regression algorithm analyzes the training data and learns the relationship between the features and the target.
The model adjusts its internal parameters to minimize the error between the predicted outcomes and the actual outcomes in the training set.




Evaluating the Model's Performance Using Accuracy Score



After training the Logistic Regression model, it's crucial to evaluate its performance to understand how well it predicts the target variable. Accuracy is one of the most common metrics used for this purpose. Here’s how you can calculate and interpret the accuracy score for both training and testing data.Calculating the accuracy score for both training and testing data provides valuable insights into the model’s performance. High accuracy on both datasets indicates a well-performing model, while discrepancies between the two can reveal issues like overfitting or underfitting. This step is essential for validating the model's reliability and effectiveness in predicting heart disease.

X_train_prediction = model.predict(X_train) : The predict method is used to make predictions on the training data (X_train).X_train_prediction contains the predicted target values for the training data.

training_data_accuracy = accuracy_score(X_train_prediction, Y_train): accuracy_score from sklearn.metrics calculates the ratio of correctly predicted instances to the total instances in the training data.training_data_accuracy represents the model’s accuracy on the training set.

X_test_prediction = model.predict(X_test) : The predict method is used to make predictions on the test data (X_test).
X_test_prediction contains the predicted target values for the test data.
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) :  accuracy_score calculates the ratio of correctly predicted instances to the total instances in the test data.
test_data_accuracy represents the model’s accuracy on the test set.




Building a predicting system



Once the Logistic Regression model has been trained and evaluated, it can be used to make predictions on new.
input_data is a tuple containing the features of a single instance. Each value corresponds to a specific medical attribute (e.g., age, sex, cholesterol level, etc.).

input_data = (57, 0, 0, 120, 354, 0, 1, 163, 1, 0.6, 2, 0, 2):input_data is a tuple containing the features of a single instance. Each value corresponds to a specific medical attribute (e.g., age, sex, cholesterol level, etc.).
input_data_as_numpy_array = np.asarray(input_data) : np.asarray(input_data) converts the input data tuple to a NumPy array. NumPy arrays are the standard input format for scikit-learn models.
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1): input_data_reshaped reshapes the array to have one row and as many columns as there are features. This is necessary because the model expects a 2D array as input, even if predicting for a single instance.

prediction = model.predict(input_data_reshaped)
print(prediction) : model.predict(input_data_reshaped) uses the trained model to predict the target value (presence or absence of heart disease) for the input data.
The result, prediction, is an array containing the predicted class (0 or 1).

if (prediction[0] == 0):
    print('Not a Heart Patient')
else:
    print('Heart Patient') : 
    
This conditional statement interprets the model's prediction.
If the predicted value (prediction[0]) is 0, it prints "Not a Heart Patient".
If the predicted value is 1, it prints "Heart Patient".












