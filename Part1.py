import numpy as np
import matplotlib.pyplot as plt  # classic plotting library
import pandas as pd  # dataframe lib
import seaborn as sns  # Modern plotting library
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

# importing the data from the internet resource
bike_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv', sep=",",
                        encoding='unicode_escape')
bike_data
logfile = PdfPages('logs.pdf')

# Taking a look at the data types
bike_data.info()

# Taking a look at the properties of the dataset
bike_data.describe()

# Finding nulls for the dataset to handle them
bike_data.isna().sum()

# finding redundant rows for the dataset
bike_data.duplicated().sum()

bike_data

# Finding the unique values for the categorical columns in order to handle them
categorical_col = ['Seasons', 'Holiday', 'Functioning Day']
result = []
for i in categorical_col:
    result.append(bike_data[i].unique())

print(result)

# Encoding the categorical values
for i in categorical_col:
    bike_data[i] = bike_data[i].astype('category')
    bike_data[i] = bike_data[i].cat.codes

# plotting the correlation matrix, refer the logs.pdf file
fig = plt.figure(figsize=(12, 10))
plt.title('Correlations')
cor = bike_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
logfile.savefig(fig)
plt.show()


# to find the correlation between the attributes
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


corr_features = correlation(bike_data, 0.85)
print(corr_features)

# Date is difficult to work with and not much useful so remove it.
del bike_data['Date']

# the function for preprocessing the dataset
def preprocess(dataset):
    # deleting the low correlation attributes
    del dataset['Dew point temperature(°C)']
    del dataset['Humidity(%)']
    del dataset['Wind speed (m/s)']
    del dataset['Visibility (10m)']
    del dataset['Rainfall(mm)']
    del dataset['Snowfall (cm)']
    del dataset['Holiday']

    # Making dataframes for training and testing the data
    sin_time = np.sin(2. * np.pi * dataset['Hour'] / max(dataset['Hour']))
    cos_time = np.cos(2. * np.pi * dataset['Hour'] / max(dataset['Hour']))
    dataset['sin_hour'] = sin_time
    dataset['cos_hour'] = cos_time

    # Getting the dataframes to be used for splitting the training and testing data
    X = dataset.iloc[:, 1:]
    Y = dataset.iloc[:, 0]

    # dividing the data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1000)

    # handling the continuous variable hours
    hour_train = X_train['Hour']
    hour_test = X_test['Hour']
    del X_train['Hour']
    del X_test['Hour']

    # scaler to normalize the data
    scaler = StandardScaler()
    X_train.iloc[:, :2] = scaler.fit_transform(X_train.iloc[:, :2])
    X_test.iloc[:, :2] = scaler.transform(X_test.iloc[:, :2])

    return X_train, X_test, Y_train, Y_test

# the class to be used 
class Gradient_Descent:

    # class constructor, inputs the X_train, Y_train datasets
    def __init__(self, X, Y):
        self.X = X
        self.y = np.array(Y).reshape(Y.shape[0],)
        self.m = X.shape[0]
        self.features = X.shape[1]

    # It
    def gradient(self, y_pred):

        weights_grad = -(2 / self.m) * (self.X.T.dot(self.y - y_pred))
        bias_grad = -(2 / self.m) * np.sum(self.y - y_pred)
        return weights_grad, bias_grad

    def gradient_descent(self, epochs, learning_rate):

        w = np.ones(shape=(self.features))
        b = 0
        cost = 0

        mse = []
        mse_list = []
        r2 = []
        r2_list = []
        cost_list = []
        epoch_list = []

        for i in range(epochs):

            y_pred = np.dot(w, self.X.T)

            weights_grad, bias_grad = self.gradient(y_pred)

            w = w - (learning_rate * weights_grad)
            b = b - (learning_rate * bias_grad)

            cost = np.mean(np.square(self.y - y_pred))

            if i % 100 == 0:
                epoch_list.append(i)
                cost_list.append(cost)

            r2_list.append(r2_score(self.y, y_pred) * 100)
            mse_list.append(mean_squared_error(self.y, y_pred))

        r2.append(np.mean(r2_list))
        mse.append(np.mean(mse_list))

        return w, b, cost, r2, mse


# applying the functions on the dataset

X_train, X_test, Y_train, Y_test = preprocess(bike_data)

grad_desc = Gradient_Descent(X_train, Y_train)

# To store the results for each iteration
w_list = {}
b_list = {}
cost_values = {}
r2_sc = {}
mse_sc = {}

# Initializing the learning_rate and epochs
learning_rate = np.arange(0.01, 0.1, 0.01)
epochs = np.arange(200, 1000, 100)

# Calling gradient_descent for multiple epochs and learning_rates on Training data
for learn_rate in learning_rate:
    for epoch in epochs:
        w, b, cost, r22, mse22 = grad_desc.gradient_descent(epoch, learn_rate)
        w_list[(np.round(learn_rate, 2), epoch)] = w
        b_list[(np.round(learn_rate, 2), epoch)] = b
        cost_values[(np.round(learn_rate, 2), epoch)] = cost
        r2_sc[(np.round(learn_rate, 2), epoch)] = r22
        mse_sc[(np.round(learn_rate, 2), epoch)] = mse22

with open('params1_log.txt' , 'w') as external_file:

    print("----------Working on Training dataset----------" , file=external_file)
    print("List of Coefficients for each iteration: ", w_list, file=external_file)
    print("List of Bias for each iteration: ", b_list, file=external_file)
    print("List of Cost for each iteration: ", cost_values, file=external_file)
    print("List of R2_Score for each iteration: ", r2_sc,  file=external_file)
    print("List of MSE for each iteration: ", mse_sc, file=external_file)


# Calling gradient_descent for multiple epochs and learning_rates on testing data

testw_list = {}
testb_list = {}
test_cost_values = {}
test_r2_sc = {}
test_mse_sc = {}
grad_desc1 = Gradient_Descent(X_test, Y_test)

learning_rate = np.arange(0.01, 0.1, 0.01)
epochs = np.arange(200, 1000, 100)

for learn_rate in learning_rate:
    for epoch in epochs:
        w, b, cost, r22, mse22 = grad_desc1.gradient_descent(epoch, learn_rate)
        testw_list[(np.round(learn_rate, 2), epoch)] = w
        testb_list[(np.round(learn_rate, 2), epoch)] = b
        test_cost_values[(np.round(learn_rate, 2), epoch)] = cost
        test_r2_sc[(np.round(learn_rate, 2), epoch)] = r22
        test_mse_sc[(np.round(learn_rate, 2), epoch)] = mse22

with open('params1_log.txt' , 'w') as external_file:
    print("----------Working on Testing dataset----------", file = external_file)
    print("List of Coefficients for each iteration: ", testw_list, file = external_file)
    print("List of Bias for each iteration: ", testb_list, file = external_file)
    print("List of Cost for each iteration: ", test_cost_values, file = external_file)
    print("List of R2_Score for each iteration: ", test_r2_sc, file = external_file)
    print("List of MSE for each iteration: ", test_mse_sc, file = external_file)


sns.set_style("darkgrid")
fig = plt.figure(figsize=(15, 12))

plt.plot(np.arange(len(learning_rate) * len(epochs)), list(cost_values.values()), color="green")
plt.xticks(np.arange(len(learning_rate) * len(epochs)), labels=cost_values.keys(), rotation='vertical')

plt.title("Training dataset Learning Rate - Epoch Vs Cost")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("Cost")
plt.tight_layout()
plt.draw()
logfile.savefig(fig)


sns.set_style("darkgrid")
fig = plt.figure(figsize=(15, 12))

plt.plot(np.arange(len(learning_rate) * len(epochs)), list(r2_sc.values()), color="green")
plt.xticks(np.arange(len(learning_rate) * len(epochs)), labels=r2_sc.keys(), rotation='vertical')

plt.title("Training dataset Learning Rate - Epoch Vs R2_Score")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("R2_Score")

plt.tight_layout()
plt.draw()

logfile.savefig(fig)

sns.set_style("darkgrid")
fig  = plt.figure(figsize=(15, 12))

plt.plot(np.arange(len(learning_rate) * len(epochs)), list(mse_sc.values()), color="green")
plt.xticks(np.arange(len(learning_rate) * len(epochs)), labels=mse_sc.keys(), rotation='vertical')

plt.title("Training dataset Learning Rate - Epoch Vs MSE")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("MSE")

plt.tight_layout()
plt.draw()

logfile.savefig(fig)

# Calculating Best value for Cost, R2_Score and MSE for training dataset

with open('params1_log.txt' , 'a') as external_file:
    print("Minimum Cost Value on Training Data: ", min(cost_values.values()) , file = external_file)
    print("Maximum R2-Score Value on Training Data: ", max(r2_sc.values()), file = external_file)
    print("Minimum MSE Value on Training Data: ", min(mse_sc.values()), file = external_file)


sns.set_style("darkgrid")
fig  = plt.figure(figsize=(15, 12))

plt.plot(np.arange(len(learning_rate) * len(epochs)), list(test_cost_values.values()), color="red")
plt.xticks(np.arange(len(learning_rate) * len(epochs)), labels=test_cost_values.keys(), rotation='vertical')

plt.title("Testing dataset Learning Rate - Epoch Vs Cost")
plt.xlabel("Learning Rate - Epoch")
plt.ylabel("Cost")

plt.tight_layout()
plt.draw()

logfile.savefig(fig)

sns.set_style("darkgrid")
fig = plt.figure(figsize=(15, 12))

plt.plot(np.arange(len(learning_rate) * len(epochs)), list(test_r2_sc.values()), color="red")
plt.xticks(np.arange(len(learning_rate) * len(epochs)), labels=test_r2_sc.keys(), rotation='vertical')

plt.title("Testing dataset Learning Rate - Epoch Vs R2_Score")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("R2_Score")

plt.tight_layout()
plt.draw()
logfile.savefig(fig)


sns.set_style("darkgrid")
fig = plt.figure(figsize=(15, 12))

plt.plot(np.arange(len(learning_rate) * len(epochs)), list(test_mse_sc.values()), color="red")
plt.xticks(np.arange(len(learning_rate) * len(epochs)), labels=test_mse_sc.keys(), rotation=90)

plt.title("Testing dataset Learning Rate - Epoch Vs MSE")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("MSE")

plt.tight_layout()
plt.draw()
logfile.savefig(fig)

# Calculating Best value Cost, R2_Score and MSE for testing dataset

with open('params1_log.txt' , 'a') as external_file:
    print("Minimum Cost Value on Testing Data: ", min(test_cost_values.values()), file = external_file)
    print("Maximum R2-Score Value on Testing Data: ", max(test_r2_sc.values()), file = external_file)
    print("Minimum MSE Value on Testing Data: ", min(test_mse_sc.values()), file = external_file)
    external_file.close()

logfile.close()
