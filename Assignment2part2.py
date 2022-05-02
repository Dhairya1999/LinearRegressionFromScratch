import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import OrdinalEncoder

dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00640/Occupancy_Estimation.csv')
dataframe.info
dataframe.columns
dataframe

logfile = PdfPages('Assignment2Logs.pdf')

class NeuralNet:

    # Define the constructor for our class
    def __init__(self, data, header=True):
        self.raw_input = pd.read_csv(data)

    # Function for preproceesing our data
    def preprocess(self):
        self.result_data = self.raw_input

        # Checking for the null  values
        print(self.result_data.isna().sum())

        # Checking for the duplicate values
        print(self.result_data.duplicated().sum())

        # Checking for the categorical columns
        print(self.result_data.select_dtypes(include=['object']).columns.tolist())

        # Finding the correlation matrix for feature selection
        fig = plt.figure(figsize=(12, 10))
        plt.title('Correlations')
        cor = self.result_data.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        logfile.savefig(fig)
        plt.show()

        # Finding out the columns that are most correlated to each other, and taking care of them
        correlated_features = set()
        for i in range(len(cor.columns)):
            for j in range(i):
                if abs(cor.iloc[i, j]) > 0.8:
                    colname = cor.columns[i]
                    correlated_features.add(colname)

        # Deleting the columns that aren't required after anaylizing the correlation matrix
        del self.result_data['Date']
        del self.result_data['Time']
        del self.result_data['S3_Light']
        del self.result_data['S4_Temp']

        cols = self.result_data.columns

        # sc = StandardScaler()
        # self.result_data = sc.fit_transform(self.result_data)
        # self.result_data = pd.DataFrame(self.result_data, columns=cols)

    def train_evaluate(self):

        # Creating the X and Y datasets for splitting them
        X = self.result_data.iloc[:, :-1]
        y = self.result_data.iloc[:, -1]
        print(X)
        print(y)

        # Creating the train and test dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

        # Finding the dimensions for our datasets
        print(X_train.shape)
        print(X_test.shape)

        # Different parameters that we will use for our nueral network and the optimizer
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 300]  # also known as epochs

        # After analysis we found that best number of layers for our neural network is 2
        num_hidden_layers = [2]

        for ac in activations:
            for lr in learning_rate:
                for hiddenlayer in num_hidden_layers:
                    for epoch in max_iterations:

                        # using the sequential
                        model = Sequential()
                        model.add(tf.keras.layers.Flatten())

                        # Adding the layers to the model, and the respective parameters
                        for i in range(hiddenlayer):
                            model.add(Dense(20, input_dim=14, activation=ac))

                        # The output layer for the model with softmax, so we know the probabilities
                        model.add(Dense(14, activation='softmax'))

                        # compiling the model, while using the adam optimizer, and calculating the metrics for accuracy
                        model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy',
                                      metrics=['accuracy'])

                        # Training the model
                        history = model.fit(X_train, y_train, epochs=epoch)

                        # Testing and evaluating the model
                        train_mse = model.evaluate(X_train, y_train)
                        test_mse = model.evaluate(X_test, y_test)

                        # The accuracy scores for our model
                        print("test Accuracy", test_mse)
                        print("train Accuracy", train_mse)

                        # Visualiziing the data for our model to compare the metrics
                        fig = plt.figure(figsize=(12, 8))
                        plot_title = "Accuracy Score vs Epoch Plot for Activation='{}', Learning Rate={}, Num_Hidden_Layers={}, Epochs={}".format(
                            ac, lr, hiddenlayer, epoch)
                        plt.title(plot_title)
                        plt.plot(history.history['accuracy'])
                        plt.xlabel("Epochs")
                        plt.ylabel("Model Accuracy")
                        plt.legend()
                        logfile.savefig(fig)
                        plt.show()



if __name__ == "__main__":
    neural_network = NeuralNet(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00640/Occupancy_Estimation.csv")
    neural_network.preprocess()
    neural_network.train_evaluate()
logfile.close()