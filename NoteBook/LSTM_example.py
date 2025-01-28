import numpy as np
from numpy import concatenate

from matplotlib import pyplot
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import plotly.graph_objects as go
import plotly.io as pio

import math, time
from math import sqrt
import random
import numpy as np
from numpy import concatenate

import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from datetime import datetime, timedelta
from matplotlib import pyplot
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
# Load the dataset
url = 'https://github.com/ahhmed207/r-colab/blob/main/Data/Machine_Learning/monthly_price.csv?raw=true'
df = pd.read_csv(url)
df.head()
# Select MONTH and Rice
df = df[['MONTH','Rice_Price']]
df.head()
# Reorder column
column_names = ['MONTH','Rice_Price']
df = df.reindex(columns=column_names)
df.head()
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='colab'


fig = go.Figure()
fig.add_traces(go.Scatter(x = df['MONTH'], y=df['Rice_Price'], mode='lines', name='Training Data'))
fig.update_layout(width=900, height=600,
                      xaxis_title = "Months", yaxis_title = "Price (US $/MT.)",
                      title = {"text": "Monthly Rice Price", "x":0.5, "pad": {'b':0, 'l':0, 't':0, 'r':0}},
                      legend = {'yanchor':'top', 'y':0.99, 'xanchor':'center', 'x':0.5})
# Display the figure
fig.show()
# Variables for training (Univariate)
cols = list(df)[1:]
price_df = df[cols].astype(float)
price_df.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(price_df)
price_scaled = scaler.transform(price_df)
### Sequence generator function

def create_sequences(data, n_future, n_past):
    X_data = []
    y_data = []
    month=[]
    month_df=df['MONTH']
    for i in range(n_past, len(price_df) - n_future +1):
        month.append(month_df[i + n_future-1:i + n_future])
        X_data.append(price_scaled[i - n_past:i, 0:price_scaled.shape[1]])
        y_data.append(price_scaled[i + n_future - 1:i + n_future, 0])

    return np.array(month), np.array(X_data), np.array(y_data)
n_future = 1   # Number of months we want to look into the future based on the past month.
n_past = 6     # Number of past months we want to use to predict the future.

month, X_data, y_data = create_sequences(price_scaled, n_future, n_past)
month.shape, X_data.shape, y_data.shape
q_80 = int(len(df['MONTH']) * .8)
q_90 = int(len(df['MONTH']) * .9)

month_train, X_train, y_train = month[:q_80], X_data[:q_80], y_data[:q_80]
month_val, X_val, y_val = month[q_80:q_90], X_data[q_80:q_90], y_data[q_80:q_90]
month_test, X_test, y_test = month[q_90:], X_data[q_90:], y_data[q_90:]

month_train.shape, X_train.shape, y_train.shape, month_val.shape, X_val.shape, y_val.shape, month_test.shape, X_test.shape, y_test.shape
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='colab'

fig = go.Figure()
fig.add_traces(go.Scatter(x = month_train.flatten(), y= y_train.flatten(), mode='markers', name='Training Data'))
fig.add_traces(go.Scatter(x = month_val.flatten(), y=y_val.flatten(), mode='markers', name='Validation Data'))
fig.add_traces(go.Scatter(x = month_test.flatten(), y=y_test.flatten(), mode='markers', name='Test Data'))
fig.update_layout(width=800, height=600,
                      xaxis_title = "Months", yaxis_title = "Std. Price",
                      title = {"text": "Training, Validation & Test Data", "x":0.5, "pad": {'b':0, 'l':0, 't':0, 'r':0}},
                      legend = {'yanchor':'top', 'y':0.99, 'xanchor':'left', 'x':0.10})
class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Xavier initialization for weights
        self.Wf = np.random.randn(hidden_dim, hidden_dim + input_dim) * np.sqrt(2.0 / (hidden_dim + (hidden_dim + input_dim)))
        self.Wi = np.random.randn(hidden_dim, hidden_dim + input_dim) * np.sqrt(2.0 / (hidden_dim + (hidden_dim + input_dim)))
        self.Wc = np.random.randn(hidden_dim, hidden_dim + input_dim) * np.sqrt(2.0 / (hidden_dim + (hidden_dim + input_dim)))
        self.Wo = np.random.randn(hidden_dim, hidden_dim + input_dim) * np.sqrt(2.0 / (hidden_dim + (hidden_dim + input_dim)))

        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))

        self.Wy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (output_dim + hidden_dim))
        self.by = np.zeros((output_dim, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.timesteps = x.shape[1]

        self.h = np.zeros((self.hidden_dim, self.batch_size, self.timesteps))
        self.c = np.zeros((self.hidden_dim, self.batch_size, self.timesteps))

        self.f = np.zeros((self.hidden_dim, self.batch_size, self.timesteps))
        self.i = np.zeros((self.hidden_dim, self.batch_size, self.timesteps))
        self.c_hat = np.zeros((self.hidden_dim, self.batch_size, self.timesteps))
        self.o = np.zeros((self.hidden_dim, self.batch_size, self.timesteps))

        self.y = np.zeros((self.output_dim, self.batch_size, self.timesteps))

        for t in range(self.timesteps):
            x_t = x[:, t].T
            h_prev = self.h[:, :, t-1] if t > 0 else np.zeros((self.hidden_dim, self.batch_size))
            c_prev = self.c[:, :, t-1] if t > 0 else np.zeros((self.hidden_dim, self.batch_size))

            concat = np.vstack((h_prev, x_t))

            self.f[:, :, t] = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            self.i[:, :, t] = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            self.c_hat[:, :, t] = self.tanh(np.dot(self.Wc, concat) + self.bc)
            self.c[:, :, t] = self.f[:, :, t] * c_prev + self.i[:, :, t] * self.c_hat[:, :, t]
            self.o[:, :, t] = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
            self.h[:, :, t] = self.o[:, :, t] * self.tanh(self.c[:, :, t])

            self.y[:, :, t] = np.dot(self.Wy, self.h[:, :, t]) + self.by

        return self.y[:, :, -1].T

    def backward(self, x, y_true, y_pred):
        m = y_true.shape[0]

        dy = (y_pred - y_true) / m
        dWy = np.dot(dy.T, self.h[:, :, -1].T)
        dby = np.sum(dy, axis=0, keepdims=True).T

        dh = np.dot(self.Wy.T, dy.T)
        do = dh * self.tanh(self.c[:, :, -1])
        dc = dh * self.o[:, :, -1] * self.dtanh(self.c[:, :, -1])

        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)

        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)

        for t in reversed(range(self.timesteps)):
            dc *= self.f[:, :, t]
            do = dh * self.tanh(self.c[:, :, t])
            df = dc * self.c[:, :, t-1] if t > 0 else np.zeros((self.hidden_dim, self.batch_size))
            di = dc * self.c_hat[:, :, t]
            dc_hat = dc * self.i[:, :, t]

            dWf += np.dot(df * self.dsigmoid(self.f[:, :, t]), np.vstack((self.h[:, :, t-1] if t > 0 else np.zeros((self.hidden_dim, self.batch_size)), x[:, t].T)).T)
            dWi += np.dot(di * self.dsigmoid(self.i[:, :, t]), np.vstack((self.h[:, :, t-1] if t > 0 else np.zeros((self.hidden_dim, self.batch_size)), x[:, t].T)).T)
            dWc += np.dot(dc_hat * self.dtanh(self.c_hat[:, :, t]), np.vstack((self.h[:, :, t-1] if t > 0 else np.zeros((self.hidden_dim, self.batch_size)), x[:, t].T)).T)
            dWo += np.dot(do * self.dsigmoid(self.o[:, :, t]), np.vstack((self.h[:, :, t-1] if t > 0 else np.zeros((self.hidden_dim, self.batch_size)), x[:, t].T)).T)

            dbf += np.sum(df, axis=1, keepdims=True)
            dbi += np.sum(di, axis=1, keepdims=True)
            dbc += np.sum(dc_hat, axis=1, keepdims=True)
            dbo += np.sum(do, axis=1, keepdims=True)

        self.Wf -= self.learning_rate * dWf
        self.Wi -= self.learning_rate * dWi
        self.Wc -= self.learning_rate * dWc
        self.Wo -= self.learning_rate * dWo

        self.bf -= self.learning_rate * dbf
        self.bi -= self.learning_rate * dbi
        self.bc -= self.learning_rate * dbc
        self.bo -= self.learning_rate * dbo

        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby

    # The fit method should be at the same indentation level as other methods
    def fit(self, X, y, epochs=100, batch_size=32):
        self.losses = []
        num_batches = X.shape[0] // batch_size

        for epoch in range(epochs):
            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                y_pred = self.forward(X_batch)
                loss = np.mean((y_pred - y_batch) ** 2)
                self.losses.append(loss)

                self.backward(X_batch, y_batch, y_pred)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        return self.forward(X)
# Create and train the LSTM model
input_dim = 1
hidden_dim = 100
output_dim = 1
learning_rate = 0.001
epochs = 200
batch_size = 32

lstm = LSTM(input_dim, hidden_dim, output_dim, learning_rate)
lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
# Training Performance
# Prediction
y_train_pred= lstm.predict(X_train)
# Invers-transformation of Predicted Data
train_predicted_copies = np.repeat(y_train_pred, price_df.shape[1], axis=-1)
train_predicted = scaler.inverse_transform(train_predicted_copies)[:,0]
# Invers-transformation of Observed Data
train_observed_copies = np.repeat(y_train, price_df.shape[1], axis=-1)
train_observed = scaler.inverse_transform(train_observed_copies)[:,0]
train_pred_df = pd.DataFrame({"Predicted":train_predicted, "Observed": train_observed})
train_pred_df["MONTHS"]=month_train.flatten()
y_train_pred.shape, train_pred_df.head()
# Vlidation Data
# Prediction
y_val_pred= lstm.predict(X_val)
# Invers-transformation of Predicted Data
val_predicted_copies = np.repeat(y_val_pred, price_df.shape[1], axis=-1)
val_predicted = scaler.inverse_transform(val_predicted_copies)[:,0]
# Invers-transformation of Observed Data
val_observed_copies = np.repeat(y_val, price_df.shape[1], axis=-1)
val_observed = scaler.inverse_transform(val_observed_copies)[:,0]
val_pred_df = pd.DataFrame({"Predicted":val_predicted, "Observed": val_observed})
val_pred_df["MONTHS"]=month_val.flatten()
val_pred_df.head()
# Test data
# Prediction
y_test_pred= lstm.predict(X_test)
# Invers-transformation of Predicted Data
test_predicted_copies = np.repeat(y_test_pred, price_df.shape[1], axis=-1)
test_predicted = scaler.inverse_transform(test_predicted_copies)[:,0]
# Invers-transformation of Observed Data
test_observed_copies = np.repeat(y_test, price_df.shape[1], axis=-1)
test_observed = scaler.inverse_transform(test_observed_copies)[:,0]
test_pred_df = pd.DataFrame({"Predicted":test_predicted, "Observed": test_observed})
test_pred_df["MONTHS"]=month_test.flatten()
test_pred_df.head()
def all_metrics(pred_df, method_name):
  import numpy as np
  # Mean Absolute Error (MAE)
  mae = np.mean(np.abs(pred_df['Predicted'].values - pred_df['Observed'].values))
  #print('MAE: %.3f' % mae)

  # Mean Absolute Percentage Error (MAPE)
  mape = np.mean(np.abs(pred_df['Predicted'].values - pred_df['Observed'].values)/pred_df['Observed'].values)
  #print('MAPE: %.3f' % mape)

  # Mean squared error (MSE)
  mse =((pred_df['Predicted'].values - pred_df['Observed'].values) ** 2).mean()
  #print('MSE: %.3f' % mse)

  # Root Mean squared error (RMSE)
  rmse =np.sqrt(((pred_df['Predicted'].values - pred_df['Observed'].values) ** 2).mean())
  #print('RMSE: %.3f' % rmse)

  # Symmetric Mean Absolute Percentage Error (SMAPE)
  smape = ((np.abs(pred_df['Observed'].values - pred_df['Predicted'].values) / (pred_df['Observed'].values + pred_df['Predicted'].values)).sum()) * (
              2.0 / pred_df['Observed'].values.size )
  #print('SMAPE: %.3f' % smape)

  # Create a pd DataFrame
  metrics_df=[{'MAE': mae, 'MAPE':  mape, 'MSE': mse, 'RMSE' : rmse, 'SMAPE': smape}]
  merge_df=pd.DataFrame(metrics_df).round(3)
  merge_df.index = [method_name]

  return merge_df
train_metrics = all_metrics(train_pred_df, method_name='Training')
val_metrics = all_metrics(val_pred_df, method_name='Validation')
test_metrics = all_metrics(test_pred_df, method_name='Test')

# combining all metrics dataframe
model_metrics = pd.concat([train_metrics, val_metrics, test_metrics])
model_metrics
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='colab'

fig = go.Figure()
fig.add_traces(go.Scatter(x = train_pred_df["MONTHS"], y = train_pred_df["Observed"], mode='markers',  name='Training Observed'))
fig.add_traces(go.Scatter(x = train_pred_df["MONTHS"],y = train_pred_df["Predicted"], mode='lines', name='Training Prediction'))
fig.add_traces(go.Scatter(x = val_pred_df["MONTHS"], y = val_pred_df["Observed"], mode='markers',  name='Validation  Observed'))
fig.add_traces(go.Scatter(x = val_pred_df["MONTHS"], y = val_pred_df["Predicted"], mode='lines', name='Validation Prediction'))
fig.add_traces(go.Scatter(x = test_pred_df["MONTHS"], y = test_pred_df["Observed"], mode='markers',  name='Test  Observed'))
fig.add_traces(go.Scatter(x = test_pred_df["MONTHS"], y = test_pred_df["Predicted"], mode='lines', name='Test Prediction'))
#fig.add_traces(go.Scatter(x = month_test.flatten(), y=y_test.flatten(), mode='lines', name='Test Data'))
fig.update_layout(width=900, height=600,
                      xaxis_title = "Months", yaxis_title = "Wheat Price (US$/Mt)",
                      title = {"text": "Global Wheat Price: Keras/TensorFlow-Univariate LSTM", "x":0.5, "pad": {'b':0, 'l':0, 't':0, 'r':0}},
                      legend = {'yanchor':'top', 'y':0.99, 'xanchor':'left', 'x':0.10})
# Load the dataset
url = 'https://github.com/ahhmed207/r-colab/blob/main/Data/Machine_Learning/monthly_price.csv?raw=true'
df = pd.read_csv(url)
# Reorder column
column_names = ['MONTH','Rice_Price', 'Crude_Oil_Price', 'Urea_Price', 'TSP_Price', 'MP_Price', 'VIX_Index']
df = df.reindex(columns=column_names)
df.head()
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='colab'
from plotly.subplots import make_subplots

fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                    vertical_spacing=0.04,
                    subplot_titles=("Rice","Crude Oil", "Urea", "TSP", "MP", "VIX-Index"),
                    y_title='Price (US $/MT)')

fig.append_trace(go.Scatter(
    x=df['MONTH'],
    y=df['Rice_Price'],
    name='Rice Price'
), row=1, col=1)

fig.append_trace(go.Scatter(
    x=df['MONTH'],
    y=df['Crude_Oil_Price'],
    name='Crude Oil Price'
), row=2, col=1)

fig.append_trace(go.Scatter(
    x= df['MONTH'],
    y=df['Urea_Price'],
    name='Urea Price'
), row=3, col=1)

fig.append_trace(go.Scatter(
    x=df['MONTH'],
    y=df['TSP_Price'],
    name='TSP Price'
), row=4, col=1)

fig.append_trace(go.Scatter(
    x=df['MONTH'],
    y=df['MP_Price'],
    name='MP Price'
), row=5, col=1)

fig.append_trace(go.Scatter(
    x=df['MONTH'],
    y=df['VIX_Index'],
    name='VIX Index'
), row=6, col=1)

fig.update_layout(width=600, height=1000, showlegend=False,
                title = {"text": "Monthly Price", "x":0.5, "pad": {'b':0, 'l':0, 't':0, 'r':0}}
                      )
fig.show()
#Variables for training
cols = list(df)[1:7]
cols
# New dataframe with only training data - 5 columns
price_df = df[cols].astype(float)
price_df.head()
def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


show_heatmap(price_df)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(price_df)
df_scaled = scaler.transform(price_df)
df_scaled.shape
#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
#In this example, the n_features is 5. We will make timesteps = 6(past months data used for training).

#Empty lists to be populated using formatted training data
X_data = []
y_data = []
month=[]
month_df=df['MONTH']

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 6  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_scaled) - n_future +1):
    X_data.append(df_scaled[i - n_past:i, 0:df_scaled.shape[1]])
    y_data.append(df_scaled[i + n_future - 1:i + n_future, 0])
    month.append(month_df[i + n_future-1:i + n_future])

X_data, y_data, month = np.array(X_data), np.array(y_data), np.array(month)
month.shape, X_data.shape, y_data.shape
np.random.seed(7)

q_80 = int(len(df['MONTH']) * .8)
q_90 = int(len(df['MONTH']) * .9)

month_train, X_train, y_train = month[:q_80], X_data[:q_80], y_data[:q_80]

month_val, X_val, y_val = month[q_80:q_90], X_data[q_80:q_90], y_data[q_80:q_90]

month_test, X_test, y_test = month[q_90:], X_data[q_90:], y_data[q_90:]

month_train.shape, X_train.shape, y_train.shape, month_val.shape, X_val.shape, y_val.shape, month_test.shape, X_test.shape, y_test.shape
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='colab'

fig = go.Figure()
fig.add_traces(go.Scatter(x = month_train.flatten(), y= y_train.flatten(), mode='markers', name='Training Data'))
fig.add_traces(go.Scatter(x = month_val.flatten(), y=y_val.flatten(), mode='markers', name='Validation Data'))
fig.add_traces(go.Scatter(x = month_test.flatten(), y=y_test.flatten(), mode='markers', name='Test Data'))
fig.update_layout(width=800, height=600,
                      xaxis_title = "Months", yaxis_title = "Std. Price",
                      title = {"text": "Training, Validation & Test Data", "x":0.5, "pad": {'b':0, 'l':0, 't':0, 'r':0}},
                      legend = {'yanchor':'top', 'y':0.99, 'xanchor':'left', 'x':0.10})
# Create and train the LSTM model
input_dim = X_train.shape[1]
hidden_dim = 100
output_dim = 1
learning_rate = 0.001
epochs = 200
batch_size = 32

lstm = LSTM(input_dim, hidden_dim, output_dim, learning_rate)
lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
# Training
# Prediction
y_train_pred= lstm.predict(X_train)
# Invers-transformation of Predicted Data
train_predicted_copies = np.repeat(y_train_pred, price_df.shape[1], axis=-1)
train_predicted = scaler.inverse_transform(train_predicted_copies)[:,0]
# Invers-transformation of Observed Data
train_observed_copies = np.repeat(y_train, price_df.shape[1], axis=-1)
train_observed = scaler.inverse_transform(train_observed_copies)[:,0]
train_pred_df = pd.DataFrame({"Predicted":train_predicted, "Observed": train_observed})
train_pred_df["MONTHS"]=month_train.flatten()
y_train_pred.shape, train_pred_df.head()
# Vlidation Data
# Prediction
y_val_pred= lstm.predict(X_val)
# Invers-transformation of Predicted Data
val_predicted_copies = np.repeat(y_val_pred, price_df.shape[1], axis=-1)
val_predicted = scaler.inverse_transform(val_predicted_copies)[:,0]
# Invers-transformation of Observed Data
val_observed_copies = np.repeat(y_val, price_df.shape[1], axis=-1)
val_observed = scaler.inverse_transform(val_observed_copies)[:,0]
val_pred_df = pd.DataFrame({"Predicted":val_predicted, "Observed": val_observed})
val_pred_df["MONTHS"]=month_val.flatten()
val_pred_df.head()
# Test data
# Prediction
y_test_pred= lstm.predict(X_test)
# Invers-transformation of Predicted Data
test_predicted_copies = np.repeat(y_test_pred, price_df.shape[1], axis=-1)
test_predicted = scaler.inverse_transform(test_predicted_copies)[:,0]
# Invers-transformation of Observed Data
test_observed_copies = np.repeat(y_test, price_df.shape[1], axis=-1)
test_observed = scaler.inverse_transform(test_observed_copies)[:,0]
test_pred_df = pd.DataFrame({"Predicted":test_predicted, "Observed": test_observed})
test_pred_df["MONTHS"]=month_test.flatten()
test_pred_df.head()
train_metrics = all_metrics(train_pred_df, method_name='Training')
val_metrics = all_metrics(val_pred_df, method_name='Validation')
test_metrics = all_metrics(test_pred_df, method_name='Test')

# combining all metrics dataframe
model_metrics = pd.concat([train_metrics, val_metrics, test_metrics])
model_metrics
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='colab'

fig = go.Figure()
fig.add_traces(go.Scatter(x = train_pred_df["MONTHS"], y = train_pred_df["Observed"], mode='markers',  name='Training Observed'))
fig.add_traces(go.Scatter(x = train_pred_df["MONTHS"],y = train_pred_df["Predicted"], mode='lines', name='Training Prediction'))
fig.add_traces(go.Scatter(x = val_pred_df["MONTHS"], y = val_pred_df["Observed"], mode='markers',  name='Validation  Observed'))
fig.add_traces(go.Scatter(x = val_pred_df["MONTHS"], y = val_pred_df["Predicted"], mode='lines', name='Validation Prediction'))
fig.add_traces(go.Scatter(x = test_pred_df["MONTHS"], y = test_pred_df["Observed"], mode='markers',  name='Test  Observed'))
fig.add_traces(go.Scatter(x = test_pred_df["MONTHS"], y = test_pred_df["Predicted"], mode='lines', name='Test Prediction'))
#fig.add_traces(go.Scatter(x = month_test.flatten(), y=y_test.flatten(), mode='lines', name='Test Data'))
fig.update_layout(width=900, height=600,
                      xaxis_title = "Months", yaxis_title = "Wheat Price (US$/Mt)",
                      title = {"text": "Global Wheat Price: Keras/TensorFlow-Multivariate LSTM", "x":0.5, "pad": {'b':0, 'l':0, 't':0, 'r':0}},
                      legend = {'yanchor':'top', 'y':0.99, 'xanchor':'left', 'x':0.10})
