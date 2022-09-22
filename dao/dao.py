import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


def main(file_path):
    # loading the diabetes dataset to a pandas DataFrame
    diabetes_dataset = pd.read_csv(file_path)

    # print shape of dataset (number of rows and columns)
    dataset_shaped = diabetes_dataset.shape
    print("dataset shaped : ", dataset_shaped)

    # print column list of dataset
    print("columns : ", list(diabetes_dataset.columns))

    # print first 5 rows of dataset
    print("data : ", diabetes_dataset.head())

    # getting the statistical measure of the data
    print("statistical data : ", diabetes_dataset.describe())

    output_value_count = diabetes_dataset["Outcome"].value_counts()
    print(output_value_count)

    # 0 is "Non-Diabetic" and 1 is "Diabetic"
    print("group of output : ", diabetes_dataset.groupby("Outcome").mean())

    # separating the data and labels
    X = diabetes_dataset.drop(columns="Outcome", axis=1)
    Y = diabetes_dataset["Outcome"]

    print("X data : ", X)
    print("Y data : ", Y)

    # Data Standardization
    scaler = StandardScaler()
    scaler.fit(X)

    standardized_data = scaler.transform(X)
    print("standardized data : ", standardized_data)

    X = standardized_data
    print("X data : ", X)

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    print(X.shape, X_train.shape, X_test.shape)

    # Train the model
    classifier = svm.SVC(kernel="linear")

    # training the support vector machine classifier
    classifier.fit(X_train, Y_train)

    # Model Evaluation
    # Accuracy score (if it is over 75, it is good accuracy)

    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print("Accuracy score of the training data : ", training_data_accuracy)  # 0.7866449511400652

    X_test_prediction = classifier.predict(X_test)
    testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print("Accuracy score of the test data : ", testing_data_accuracy)  # 0.7727272727272727

    # test model prediction
    input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31)

    # changing the input data to numpy array
    input_data_as_np_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_np_array.reshape(1, -1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)
    print("predicted data : ", prediction)

    if prediction[0] == 0:
        return "The person is not diabetic."
    else:
        return "The person is diabetic."
