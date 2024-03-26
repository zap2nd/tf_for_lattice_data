import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_file_arr = np.loadtxt("lat_GPD.txt", skiprows=1, usecols=0) #import data from file
minus_t_arr = np.loadtxt("lat_GPD.txt", skiprows=1, usecols=1)
H_arr = np.loadtxt("lat_GPD.txt", skiprows=1, usecols=2)
H_err_arr = np.loadtxt("lat_GPD.txt", skiprows=1, usecols=3)

#need to get H as a function of x for each value of -t (= Q^2)
x_arr = [] #start by declaring as a np array
pattern_indices = np.arange(0, len(x_file_arr), 5)
x_arr = np.append(x_arr, x_file_arr[pattern_indices])

num_minus_t = 5
H_arr_all_t = np.array([[]])
H_err_arr_all_t = np.array([[]])
minust_vals_arr = []

for i in np.arange(num_minus_t):
    minust_vals_arr.append(minus_t_arr[i])

for i in np.arange(num_minus_t): 
    pattern_indices = np.arange(i, len(H_arr), 5)
    H_arr_all_t = np.append(H_arr_all_t, H_arr[pattern_indices])
    H_err_arr_all_t = np.append(H_err_arr_all_t, H_err_arr[pattern_indices])

H_vals_t = H_arr_all_t[:16]
H_err_vals_t = H_err_arr_all_t[:16]

x_train = x_arr[10: 16]
y_train = H_vals_t[10: 16]
err_train = H_err_vals_t[10: 16]
x_test = x_arr[0: 10]
y_test = H_vals_t[0: 10]
err_test = H_err_vals_t[0: 10]

#Define a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=20, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=1)
    ])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(np.log(x_train), np.log(y_train), epochs=1000, verbose=1)

x_pred = np.log(np.linspace(0.1, 1, 100))
y_pred = model.predict(x_pred)

plt.scatter(x_train, y_train, label='Training data', marker='o')
plt.scatter(x_test, y_test, label='Testing data', marker='.')
plt.plot(np.exp(x_pred), np.exp(y_pred), color='red', label='Extrapolated prediction')
plt.xlabel('X')
plt.ylabel(r'$H(X, \zeta = 0, t = 0)$')
plt.title('Neural network attempt - Training on log of the data')
plt.legend()
plt.yscale('log')
plt.savefig('nn_ex.png')
