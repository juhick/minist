import input_data
import train
import numpy as np

minist = input_data.read_data_sets("data/", one_hot=True)



train_set_x_orig = np.array(minist.train.images[:])
train_set_y = np.array(minist.train.labels[:])
test_set_x_orig = np.array(minist.test.images[:])
test_set_y = np.array(minist.test.labels[:])

print(train_set_x_orig.shape)
print(train_set_y.shape)

train_set_y = train_set_y.reshape((10, train_set_y.shape[0]))
test_set_y = test_set_y.reshape((10, test_set_y.shape[0]))

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print(train_x_flatten.shape)

#train_x = train_x_flatten
train_x = train_x_flatten / 255
train_y = train_set_y
#test_x = test_x_flatten
test_x = test_x_flatten / 255
test_y = test_set_y

layers_dims = [784, 50, 40, 20, 10]

n_x = 784
n_h = 50
n_y = 10

parameters = train.two_layer_model(train_x, train_set_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True,isPlot=True)
#parameters = train.L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True, isPlot=True)