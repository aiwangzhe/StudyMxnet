import gluonbook as gb
from mxnet import nd


batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)