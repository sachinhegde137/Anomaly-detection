import numpy as np
import tensorflow as tf
import pandas as pd
import os
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed


rcParams['figure.figsize'] = 14, 8
np.random.seed(1)
tf.random.set_seed(1)
print('Tensorflow version:', tf.__version__)

def plot_data(x, y=None, type='line', xlabel=None, ylabel=None, title=None, bins=None):
    plt.xlabel(xlabel, fontsize='large')
    plt.ylabel(ylabel, fontsize='large')
    plt.title(title, fontsize=14)
    #plt.legend(legend, loc='upper left')

    if type == 'line':
        plt.plot(x,y)
    elif type == 'scatter':
        plt.scatter(x,y)
    elif type == 'hist':
        plt.hist(x, bins=bins)

def create_dataset(inp, outp, time_steps=1):
    Xs, ys = [], []
    for i in range(len(inp) - time_steps):
        v = inp.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(outp.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


df = pd.read_csv('data\\ambient_temperature_system.csv', parse_dates=['timestamp'])
print("The first 5 entries of the sensor data: {}".format(df.head()))
print("The shape of the dataframe: {}".format(df.shape))


output_files_path = "results"
if not os.path.exists(output_files_path):
    os.makedirs(output_files_path)

''' Plot sensor data over time '''
plt.figure()
plot_data(x=df.timestamp, y=df.value, type='line', xlabel='timestamp',
          ylabel='data', title='Sensor data over time')
output_path = output_files_path + "/sensor_data_over_time.png"
plt.savefig(output_path)
plt.close()

''' Plot histogram to understand the distribution '''
plt.figure()
plot_data(x=df.value, type='hist', xlabel='sensor value',
          ylabel='number of data points', title='Histogram', bins=10)
output_path = output_files_path + "/Histogram.png"
plt.savefig(output_path)
plt.close()


train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size].copy(), df.iloc[train_size:len(df)].copy()
print("Size of training data: {}".format(train.shape))
print("Size of test data: {}".format(test.shape))

scaler = StandardScaler()
scaler = scaler.fit(train[['value']])

train['value'] = scaler.transform(train[['value']])
test['value'] = scaler.transform(test[['value']])

''' Plot train and test data over time '''
plt.figure()
plot_data(x=train.timestamp, y=train.value, type='line')
plot_data(x=test.timestamp, y=test.value, type='line', xlabel='timestamp',
          ylabel='data', title='Data after scaling and split into train and test dataset')
output_path = output_files_path + "/scaled_data.png"

plt.legend(['train data', 'test data'], loc='upper left')
plt.savefig(output_path)
plt.close()


time_steps = 24
X_train, y_train = create_dataset(train[['value']], train.value, time_steps)
X_test, y_test = create_dataset(test[['value']], test.value, time_steps)

print("the shape of training data: {}".format(X_train.shape))
print("the shape of test data: {}".format(X_test.shape))

timesteps = X_train.shape[1]
num_features = X_train.shape[2]

model = Sequential([
    LSTM(128, input_shape=(timesteps, num_features)),
    Dropout(0.2),
    RepeatVector(timesteps),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(num_features))
])

model.compile(loss='mae', optimizer='adam')
model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks = [es, mc],
    shuffle=False
)

#Load the best model observed as saved_model
saved_model = tf.keras.models.load_model('best_model.h5')

''' Plot training and validation loss data over time '''
plt.figure()
num_of_epochs = len(history.history['loss'])
x_epochs = [i for i in range(num_of_epochs)]
plot_data(x=x_epochs, y=history.history['loss'], type='line')
plot_data(x=x_epochs, y=history.history['val_loss'], type='line', xlabel='epoch',
          ylabel='loss', title='Loss summary')
output_path = output_files_path + "/loss_summary.png"
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(output_path)
plt.close()

X_train_pred = saved_model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
saved_model.evaluate(X_test, y_test)

''' Plot histogram of mean absolute error of training data '''
plt.figure()
plot_data(x=train_mae_loss, type='hist', xlabel='mae of training data',
          ylabel='number of data points', title='train_mae_loss', bins=50)
output_path = output_files_path + "/train_mae_loss.png"
plt.savefig(output_path)
plt.close()

X_test_pred = saved_model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

''' Plot histogram of mean absolute error of test data '''
plt.figure()
plot_data(x=test_mae_loss, type='hist', xlabel='mae of test data',
          ylabel='number of data points', title='test_mae_loss', bins=50)
output_path = output_files_path + "/test_mae_loss.png"
plt.savefig(output_path)
plt.close()

threshold = np.max(train_mae_loss)
threshold_1 = 1.5

test_score_df = pd.DataFrame(test[time_steps:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['value'] = test[time_steps:].value

print(test_score_df.head())

''' Scatter plot of loss vs timestamp to see the distribution '''
plt.figure()
plot_data(x=test[time_steps:].timestamp, y=test_score_df.loss, type='scatter')
plot_data(x=test[time_steps:].timestamp, y=test_score_df.threshold, type='scatter', xlabel='timestamp',
          ylabel='loss', title='Test Loss')
output_path = output_files_path + "/test_loss.png"
plt.legend(['loss', 'threshold'], loc='upper left')
plt.savefig(output_path)
plt.close()

anomalies = test_score_df[test_score_df.anomaly == True]
print(anomalies.head())

''' Scatter plot of sensor vs timestamp to see the anomalies '''
plt.figure()
plot_data(x=test[time_steps:].timestamp, y=scaler.inverse_transform(test[timesteps:].value),
          type='scatter')
plot_data(x=anomalies.timestamp, y=scaler.inverse_transform(anomalies.value),
          type='scatter', xlabel='timestamp', ylabel='test data',
          title='Normal data vs Anomalies')
output_path = output_files_path + "/Normal data vs Anomalies.png"
plt.legend(['anomalies', 'normal data'], loc='upper left')
plt.savefig(output_path)
plt.close()