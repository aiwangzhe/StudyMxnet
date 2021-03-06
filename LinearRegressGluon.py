from mxnet import autograd, nd

num_inputs = 2
num_examples = 10000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

from mxnet.gluon import data as gdata

batch_size = 10
# 将训练数据的特征和标签组合。
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量。
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(1))

from mxnet import init

net.initialize(init.Normal(sigma=0.01))

from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()  # 平方损失又称 L2 范数损失。

from mxnet import gluon

print(net.collect_params())

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())