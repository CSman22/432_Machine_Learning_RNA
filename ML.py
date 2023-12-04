"""
Author      : Jackey Weng
Student ID  : 40130001
Description : Assignment 2
"""
import os
import pandas as pd
import numpy as np
from Model.RNA import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import statistics
import matplotlib.pyplot as plt


class Machine_Learning:
    def __init__(self, dataset: str):
        self.dataset = dataset

    @staticmethod
    def one_hot_vector(sequence):
        """Convert the RNA sequence to a list of integers"""
        # Assign each nucleotide to a number
        mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
        # Map each nucleotide in the sequence to the respective number
        seq2 = [mapping[i] for i in sequence]
        return seq2

    def dataset_processing(self):
        """Return a list of each instance from the text file"""
        current_directory = os.getcwd()
        data_path = os.path.join(current_directory, self.dataset)
        line_index = 0
        rna_list = []

        with open(data_path) as file:
            for line in file:
                line_index += 1
                # Preprocessing the data
                # Split the instance into sequence, BPPM and activity level
                data = line.split(";")

                # convert the sequence to one hot vector
                sequence = data[1]
                sequence = self.one_hot_vector(sequence)

                # Converting the list of string to a list of float
                # remove the brackets and single quotes
                probability_str = data[2].strip("[]").replace("'", "")
                # convert the string to a list of float
                probability_list = [
                    float(value) for value in probability_str.split(",")
                ]

                # Get the upper triangle of the matrix
                upper_triangle = []
                matrix_size = 60
                for i in range(matrix_size):
                    for j in range(i + 1, matrix_size):
                        # calculate the index to start retrieving the upper triangle values
                        index = i * matrix_size + j
                        upper_triangle.append(probability_list[index])

                # print(upper_triangle)
                # Get the activity level
                activity_level = float(data[3])

                # Create an instance of an RNA
                rna = RNA(sequence, upper_triangle, activity_level)
                rna_list.append(rna)

        return rna_list

    def train_test_set(self, rna_list, test_size: float = None):
        """
        Split the dataset into training and testing sets

        Parameters:
            rna_list (list): A list of RNA objects
            test_size (float): The size of the testing set [0,1]
        Returns:
            x_train (list): A list of the training features
            x_test (list): A list of the testing features
            y_train (list): A list of the training target
            y_test (list): A list of the testing target
        """
        # Convert the list of RNA instances to a dictionary
        dataset = [rna.to_dictionary() for rna in rna_list]

        # Convert the dictionary to a dataframe
        df = pd.DataFrame(dataset)

        # Get only the features
        x = df.drop(columns=["activity_level"])

        # Get the target
        y = df["activity_level"]

        # Split the dataset into training and testing sets
        # Setting the random state to 1 will ensure the same split every time
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=1
        )
        return x_train, x_test, y_train, y_test

    def train_knn(self, x_train, x_test, y_train, y_test):
        """
        Modelling: Build KNN, evaluate and optimize model

        Parameters:
            x_train (list): A list of the training features
            x_test (list): A list of the testing features
            y_train (list): A list of the training target
            y_test (list): A list of the testing target

        Returns:
        """
        # Create a KNN regressor model
        knn = KNeighborsRegressor()

        # Create a dictionary of the parameters to be tuned
        params_dict = {
            "n_neighbors": np.arange(1, 25),
        }

        # Create a grid search to test all the n neighbors while using 10 fold cross validation
        knn_cv = GridSearchCV(knn, params_dict, cv=10, scoring="neg_mean_squared_error")

        # Fit the model to the training data
        knn_cv.fit(x_train, y_train)
        print(f"Best n_neighbors: {knn_cv.best_params_}")

        # Get the best knn model from the grid search
        best_knn = knn_cv.best_estimator_

        # Make predictions on the testing set
        predictions = best_knn.predict(x_test)

        # Calculate the MSE, MAE and R2
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"R2 : {r2}")

        # print(predictions[1:10])
        # print("----")
        # print(y_test[1:10])
        return predictions

    # TODO Create graphs
    def visualize(self, predictions, y_test):
        """
        Create a scatter plot of the predicted values vs the actual values
        Create a scatter plot of the residuals vs the actual values
        """
        # Create 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        # Scatter plot of predicted values vs the actual values
        axes[0].scatter(
            y_test,
            predictions,
            alpha=0.7,
            c=(abs(y_test - predictions)),
            cmap="viridis",
        )
        axes[0].set_title("Predicted vs Actual Values")
        axes[0].set_xlabel("Actual Values")
        axes[0].set_ylabel("Predicted Values")

        # Residual plot
        residuals = y_test - predictions
        axes[1].scatter(
            y_test, residuals, alpha=0.7, c=(abs(residuals)), cmap="viridis"
        )
        axes[1].set_title("Residual Plot")
        axes[1].set_xlabel("Actual Values")
        axes[1].set_ylabel("Residuals")
        axes[1].axhline(y=0, color="g", linestyle="--")

        # Adjust layout to prevent overlap
        # plt.tight_layout()

        plt.show()

        graph_path = os.path.join(os.getcwd(), "analysis.png")
        fig.savefig(graph_path, format="png")


def main():
    # Create an instance of the Machine Learning class
    ml = Machine_Learning(dataset="Data/A2_15000.txt")

    # Step 1: Preprocessing the dataset
    dataset = ml.dataset_processing()

    # Step 2: Split data set into training and testing (x= feature, y = target)
    # With the test size of 20% the training will be 12,000 and testing will be 3,000 instances
    x_train, x_test, y_train, y_test = ml.train_test_set(dataset, test_size=0.2)

    # Step 3: Modelling the knn
    predictions = ml.train_knn(x_train, x_test, y_train, y_test)

    # Step 4: Visualize the results
    ml.visualize(predictions, y_test)


if __name__ == "__main__":
    main()
