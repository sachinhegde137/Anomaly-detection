# Anomaly-detection in Ambient temperature system
## Objective 

Main goal of this project is to understand the algorithms used to for time series data analysis. As an overview, the project contains two main taasks.
1. Development of LSTM Autoencoder in Keras
2. Design an anomaly detection system for anomalies in ambient temperature system in office settings

## Dataset

The data is stored in .csv format and is obtained from [The Numenta Anomaly Benchmark (NAB) repository](https://github.com/numenta/NAB). The .csv file in the *data* directory
contains the temperature sensor data, recorded every hour, between 04/07/2013 and 28/05/2014. The .csv file has two columns"timestamp" and "value". These values are used to
train an LSTM Autoencoder to predict anomalies in the values. For training, 80% of the data is used and the rest of the data is used for testing. The figure below shows the sensor 
values over time.
![Sensor data over time](https://github.com/sachinhegde137/Anomaly-detection/blob/master/results/sensor_data_over_time.png)
As shown in the Histogram of the sensor data in the figure below, it is observed the data fits a Guassian distribution with a well behaved mean and standard deviation.
The data is standardized and split into train and test data. Standardizing rescales the data in such a way that the distribution of values have mean of 0 and standard deviation of 1.
![Histogram](https://github.com/sachinhegde137/Anomaly-detection/blob/master/results/Histogram.png)
![Standardized data](https://github.com/sachinhegde137/Anomaly-detection/blob/master/results/scaled_data.png)

## Requirements
+ Tensorflow 2.3.0
+ NumPy 1.19.3
+ Pandas 1.0.1
+ Keras 2.3.1
+ Matplotlib 3.1.2 
+ Scikit Learn 0.23.2

## Proposed approach
Long Short-Term Memory, or LSTM, networks are specifically designed to make sense of sequence of data. They are capable of storing information and use it across long input sequences.
The Encoder-Decoder LSTM architecture allows the model to be used on variable length input sequences and to predict the variable length output sequences. 

An LSTM Autoencoder is an implementation of an autoencoder for sequence data using an Encoder-Decoder LSTM architecture. For a given sequence of input data, an Encoder-Decoder LSTM 
model is configured to encode, decode and recreate the data. The performance of the model is evaluated based on the model’s ability to recreate the input sequence.
LSTM network expects the input in the shape *[samples, time_steps, features]*. Since the data is recorded every hour, we can configure the network
to have the memory of previous day's data which is 24 values. This makes LSTM Autoencoder takes the input sequences of 30 time steps and 1 feature and outputs a sequence of 30 time steps
and 1 feature. Once the model is trained, *mean absolute error (mae)* is calculated on training data. The maximum value of *mae* of training data is set as threshold for reconstruction error.
If the *mae* value of a data point in the test data is above this threshold, then that data point is considered as an anomaly. 

## Results
Once all the dependencies are installed, we can run the anomaly detection python file. To run the the anomaly detection on the sensor data,
download the repository and run `python anomaly_detect.py` in the command window from the project directory. This will create a folder 'results' and plot the loss summary,
mean absolute errors of train and test data, and scatter plot of normal data vs anomalies. All the plots are shown below.
![Loss summary](https://github.com/sachinhegde137/Anomaly-detection/blob/master/results/loss_summary.png)
![Test loss and threshold](https://github.com/sachinhegde137/Anomaly-detection/blob/master/results/test_loss.png)
![Normal data vs Anomalies](https://github.com/sachinhegde137/Anomaly-detection/blob/master/results/Normal%20data%20vs%20Anomalies.png)

## References
+ Jason Brownlee, [A Gentle Introduction to LSTM Autoencoders](https://machinelearningmastery.com/lstm-autoencoders/)
+ [The Numenta Anomaly Benchmark (NAB) repository](https://github.com/numenta/NAB)
+ Pankaj Malhotra et al., "Long Short Term Memory Networks for Anomaly Detection in Time Series", ESANN 2015 proceedings, European Symposium on Artificial Neural Networks,
Computational Intelligence and Machine Learning, Bruges (Belgium), 22-24 April 2015. [PDF](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56.pdf)
