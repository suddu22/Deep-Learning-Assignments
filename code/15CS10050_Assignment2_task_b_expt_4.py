import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data import dataset
from data_loader import DataLoader
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import random
import sys


args = sys.argv[1]

ctx = mx.cpu()
dl = DataLoader()

batch_size = 64
num_inputs = 784
num_outputs = 10

class CustomDataset(dataset.Dataset):
    def __init__(self, mode, val):
        self.X, self.y = dl.load_data(mode, val)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


# custom function to perform batch sgd
def sgd(params, lr, batch_size):
    for p in params:
        p[:] = p - lr * p.grad / batch_size


# custom function to perform nesterov's accelerated gradient descent
def nesterov_momentum(params, vs, lr, mom, batch_size):
    for param, v in zip(params, vs):
        v[:] = mom * v + lr * param.grad / batch_size
        param[:] = param - v



def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = mx.ndarray.cast(data, dtype='float32')
        data = data.as_in_context(ctx).reshape((-1, num_inputs))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

num_hidden2 = [1024, 512, 256]
net_sgd = gluon.nn.Sequential()
with net_sgd.name_scope():
    net_sgd.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net_sgd.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net_sgd.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net_sgd.add(gluon.nn.Dense(num_outputs))

net_nam = gluon.nn.Sequential()
with net_nam.name_scope():
    net_nam.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net_nam.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net_nam.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net_nam.add(gluon.nn.Dense(num_outputs))


net_adadelta = gluon.nn.Sequential()
with net_adadelta.name_scope():
    net_adadelta.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net_adadelta.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net_adadelta.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net_adadelta.add(gluon.nn.Dense(num_outputs))

net_adagrad = gluon.nn.Sequential()
with net_adagrad.name_scope():
    net_adagrad.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net_adagrad.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net_adagrad.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net_adagrad.add(gluon.nn.Dense(num_outputs))

net_rmsprop = gluon.nn.Sequential()
with net_rmsprop.name_scope():
    net_rmsprop.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net_rmsprop.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net_rmsprop.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net_rmsprop.add(gluon.nn.Dense(num_outputs))


num_hidden2 = [1024, 512, 256]
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2.add(gluon.nn.Dense(num_outputs))

net2.load_parameters("../weights/net2.params", ctx=ctx)

optimizers = ['adadelta', 'adagrad', 'rmsprop']


if args == '--train':

    epochs = 10

    train_dataset = CustomDataset('train', val=False)
    train_loader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = CustomDataset('train', val=True)
    val_loader = mx.gluon.data.DataLoader(val_dataset, batch_size, shuffle=False)

    num_examples = len(train_dataset)


    print('Network 2 - SGD')

    net_sgd_pts = []
    net_sgd_val = []

    net_sgd.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    

    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net_sgd(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            weights = [] # for passing weights to sgd function
            for i in range(len(num_hidden2) + 1):
                weights.append(net_sgd.__getitem__(i).weight.data())
                weights.append(net_sgd.__getitem__(i).bias.data())
            sgd(weights, 0.001, batch_size)
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net_sgd)
        train_accuracy = evaluate_accuracy(train_loader, net_sgd)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net_sgd_pts.append(cumulative_loss/num_examples)
        net_sgd_val.append(val_accuracy)

    file_name = "../weights/net_sgd.params"
    net_sgd.save_parameters(file_name)


    print('Network 2 - Nesterov\'s Accelerated Momentum')

    net_nam.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    
    vs = [] # velocity in momentum based update
    
    # vs.append(mx.nd.zeros(net_nam._children['0']._params._params['sequential1_dense0_weight'].shape[::-1]))
    # vs.append(mx.nd.zeros(net_nam._children['0']._params._params['sequential1_dense0_bias'].shape))

    # vs.append(mx.nd.zeros(net_nam._children['1']._params._params['sequential1_dense1_weight'].shape[::-1]))
    # vs.append(mx.nd.zeros(net_nam._children['1']._params._params['sequential1_dense1_bias'].shape))

    # vs.append(mx.nd.zeros(net_nam._children['2']._params._params['sequential1_dense2_weight'].shape[::-1]))
    # vs.append(mx.nd.zeros(net_nam._children['2']._params._params['sequential1_dense2_bias'].shape))

    # initialize vs with all zeros
    vs.append(mx.nd.zeros((num_hidden2[0], num_inputs)))
    vs.append(mx.nd.zeros(num_hidden2[0]))

    vs.append(mx.nd.zeros((num_hidden2[1], num_hidden2[0])))
    vs.append(mx.nd.zeros(num_hidden2[1]))

    vs.append(mx.nd.zeros((num_hidden2[2], num_hidden2[1])))
    vs.append(mx.nd.zeros(num_hidden2[2]))

    net_nam_pts = []
    net_nam_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net_nam(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            weights = []
            for i in range(len(num_hidden2) + 1):
                weights.append(net_nam.__getitem__(i).weight.data())
                weights.append(net_nam.__getitem__(i).bias.data())
            nesterov_momentum(weights, vs, 0.001, 0.9, batch_size)
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net_nam)
        train_accuracy = evaluate_accuracy(train_loader, net_nam)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net_nam_pts.append(cumulative_loss/num_examples)
        net_nam_val.append(val_accuracy)

    file_name = "../weights/net_nam.params"
    net_nam.save_parameters(file_name)

    print('Network 2 - AdaDelta')

    net_adadelta.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net_adadelta.collect_params(), 'adadelta', {'learning_rate': .001})

    
    net_adadelta_pts = []
    net_adadelta_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net_adadelta(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net_adadelta)
        train_accuracy = evaluate_accuracy(train_loader, net_adadelta)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net_adadelta_pts.append(cumulative_loss/num_examples)
        net_adadelta_val.append(val_accuracy)

    file_name = "../weights/net_adadelta.params"
    net_adadelta.save_parameters(file_name)


    print('Network 2 - AdaGrad')

    net_adagrad.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net_adagrad.collect_params(), 'adagrad', {'learning_rate': .001})

    
    net_adagrad_pts = []
    net_adagrad_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net_adagrad(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net_adagrad)
        train_accuracy = evaluate_accuracy(train_loader, net_adagrad)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net_adagrad_pts.append(cumulative_loss/num_examples)
        net_adagrad_val.append(val_accuracy)

    file_name = "../weights/net_adagrad.params"
    net_adagrad.save_parameters(file_name)


    print('Network 2 - RMSProp')

    net_rmsprop.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net_rmsprop.collect_params(), 'rmsprop', {'learning_rate': .001})

    
    net_rmsprop_pts = []
    net_rmsprop_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net_rmsprop(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net_rmsprop)
        train_accuracy = evaluate_accuracy(train_loader, net_rmsprop)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net_rmsprop_pts.append(cumulative_loss/num_examples)
        net_rmsprop_val.append(val_accuracy)

    file_name = "../weights/net_rmsprop.params"
    net_rmsprop.save_parameters(file_name)


    net2_pts = []
    net2_val = []

    with open("../weights/net2_pts.txt", "r") as f:
        for line in f:
            net2_pts.append(float(line.strip()))

    with open("../weights/net2_val.txt", "r") as f:
        for line in f:
            net2_val.append(float(line.strip()))


    x = range(0, len(net2_pts))
    fig, ax = plt.subplots()
    ax.plot(x, net_sgd_pts, label='Network2_SGD')
    ax.plot(x, net_nam_pts, label='Network2_NAM')
    ax.plot(x, net_adadelta_pts, label='Network2_AdaDelta')
    ax.plot(x, net_adagrad_pts, label='Network2_AdaGrad')
    ax.plot(x, net_rmsprop_pts, label='Network2_RMSProp')
    ax.plot(x, net2_pts, label='Network2_Adam')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(top=5)
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/loss_task_b_expt_4.png')


    x = range(0, len(net2_val))
    fig, ax = plt.subplots()
    ax.plot(x, net_sgd_val, label='Network2_SGD')
    ax.plot(x, net_nam_val, label='Network2_NAM')
    ax.plot(x, net_adadelta_val, label='Network2_AdaDelta')
    ax.plot(x, net_adagrad_val, label='Network2_AdaGrad')
    ax.plot(x, net_rmsprop_val, label='Network2_RMSProp')
    ax.plot(x, net2_val, label='Network2_Adam')
    plt.xlabel('epoch')
    plt.ylabel('val_accuracy')
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/val_task_b_expt_4.png')



if args == '--test':

    test_dataset = CustomDataset('test', val=False)
    test_loader = mx.gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)
    
    net_sgd.load_parameters("../weights/net_sgd.params", ctx=ctx)
    test_accuracy = evaluate_accuracy(test_loader, net_sgd)
    print("Test Accuracy of Network 2 - SGD: " + str(test_accuracy))

    net_nam.load_parameters("../weights/net_nam.params", ctx=ctx)
    test_accuracy = evaluate_accuracy(test_loader, net_nam)
    print("Test Accuracy of Network 2 - Nesterov's Accelerated Momentum: " + str(test_accuracy))

    net_adadelta.load_parameters("../weights/net_adadelta.params", ctx=ctx)
    test_accuracy = evaluate_accuracy(test_loader, net_adadelta)
    print("Test Accuracy of Network 2 - AdaDelta: " + str(test_accuracy))

    net_adagrad.load_parameters("../weights/net_adagrad.params", ctx=ctx)
    test_accuracy = evaluate_accuracy(test_loader, net_adagrad)
    print("Test Accuracy of Network 2 - AdaGrad: " + str(test_accuracy))

    net_rmsprop.load_parameters("../weights/net_rmsprop.params", ctx=ctx)
    test_accuracy = evaluate_accuracy(test_loader, net_rmsprop)
    print("Test Accuracy of Network 2 - RMSProp: " + str(test_accuracy))

    net2.load_parameters("../weights/net2.params", ctx=ctx)
    test_accuracy = evaluate_accuracy(test_loader, net2)
    print("Test Accuracy of Network 2 - Adam: " + str(test_accuracy))