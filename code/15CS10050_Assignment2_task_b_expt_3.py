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
net2_dropout_01 = gluon.nn.Sequential()
with net2_dropout_01.name_scope():
    net2_dropout_01.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2_dropout_01.add(gluon.nn.Dropout(.1))
    net2_dropout_01.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2_dropout_01.add(gluon.nn.Dropout(.1))
    net2_dropout_01.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2_dropout_01.add(gluon.nn.Dropout(.1))
    net2_dropout_01.add(gluon.nn.Dense(num_outputs))

net2_dropout_04 = gluon.nn.Sequential()
with net2_dropout_04.name_scope():
    net2_dropout_04.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2_dropout_04.add(gluon.nn.Dropout(.4))
    net2_dropout_04.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2_dropout_04.add(gluon.nn.Dropout(.4))
    net2_dropout_04.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2_dropout_04.add(gluon.nn.Dropout(.4))
    net2_dropout_04.add(gluon.nn.Dense(num_outputs))

net2_dropout_06 = gluon.nn.Sequential()
with net2_dropout_06.name_scope():
    net2_dropout_06.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2_dropout_06.add(gluon.nn.Dropout(.6))
    net2_dropout_06.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2_dropout_06.add(gluon.nn.Dropout(.6))
    net2_dropout_06.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2_dropout_06.add(gluon.nn.Dropout(.6))
    net2_dropout_06.add(gluon.nn.Dense(num_outputs))



num_hidden2 = [1024, 512, 256]
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2.add(gluon.nn.Dense(num_outputs))

net2.load_parameters("../weights/net2.params", ctx=ctx)

if args == '--train':

    train_dataset = CustomDataset('train', val=False)
    train_loader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = CustomDataset('train', val=True)
    val_loader = mx.gluon.data.DataLoader(val_dataset, batch_size, shuffle=False)

    num_examples = len(train_dataset)

    print('Network 2 - Dropout = 0.1')

    net2_dropout_01.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net2_dropout_01.collect_params(), 'adam', {'learning_rate': .001})

    epochs = 10
    net2_dropout_01_pts = []
    net2_dropout_01_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net2_dropout_01(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net2_dropout_01)
        train_accuracy = evaluate_accuracy(train_loader, net2_dropout_01)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net2_dropout_01_pts.append(cumulative_loss/num_examples)
        net2_dropout_01_val.append(val_accuracy)

    file_name = "../weights/net2_dropout_01.params"
    net2_dropout_01.save_parameters(file_name)


    print('Network 2 - Dropout = 0.4')


    net2_dropout_04.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net2_dropout_04.collect_params(), 'adam', {'learning_rate': .001})

    net2_dropout_04_pts = []
    net2_dropout_04_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net2_dropout_04(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net2_dropout_04)
        train_accuracy = evaluate_accuracy(train_loader, net2_dropout_04)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net2_dropout_04_pts.append(cumulative_loss/num_examples)
        net2_dropout_04_val.append(val_accuracy)

    file_name = "../weights/net2_dropout_04.params"
    net2_dropout_04.save_parameters(file_name)


    print('Network 2 - Dropout = 0.6')


    net2_dropout_06.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net2_dropout_06.collect_params(), 'adam', {'learning_rate': .001})

    net2_dropout_06_pts = []
    net2_dropout_06_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net2_dropout_06(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()

        val_accuracy = evaluate_accuracy(val_loader, net2_dropout_06)
        train_accuracy = evaluate_accuracy(train_loader, net2_dropout_06)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net2_dropout_06_pts.append(cumulative_loss/num_examples)
        net2_dropout_06_val.append(val_accuracy)

    file_name = "../weights/net2_dropout_06.params"
    net2_dropout_06.save_parameters(file_name)


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
    ax.plot(x, net2_dropout_01_pts, label='Network2_Dropout=0.1')
    ax.plot(x, net2_dropout_04_pts, label='Network2_Dropout=0.4')
    ax.plot(x, net2_dropout_06_pts, label='Network2_Dropout=0.6')
    ax.plot(x, net2_pts, label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(top=5)
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/loss_task_b_expt_3.png')


    x = range(0, len(net2_val))
    fig, ax = plt.subplots()
    ax.plot(x, net2_dropout_01_val, label='Network2_Dropout=0.1')
    ax.plot(x, net2_dropout_04_val, label='Network2_Dropout=0.4')
    ax.plot(x, net2_dropout_06_val, label='Network2_Dropout=0.6')
    ax.plot(x, net2_val, label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('val_accuracy')
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/val_task_b_expt_3.png')



if args == '--test':
    
    net2_dropout_01.load_parameters("../weights/net2_dropout_01.params", ctx=ctx)
    net2_dropout_04.load_parameters("../weights/net2_dropout_04.params", ctx=ctx)
    net2_dropout_06.load_parameters("../weights/net2_dropout_06.params", ctx=ctx)
    net2.load_parameters("../weights/net2.params", ctx=ctx)

    test_dataset = CustomDataset('test', val=False)
    test_loader = mx.gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)
    
    test_accuracy = evaluate_accuracy(test_loader, net2_dropout_01)
    print("Test Accuracy of Network 2 - Dropout = 0.1: " + str(test_accuracy))

    test_accuracy = evaluate_accuracy(test_loader, net2_dropout_04)
    print("Test Accuracy of Network 2 - Dropout = 0.4: " + str(test_accuracy))

    test_accuracy = evaluate_accuracy(test_loader, net2_dropout_06)
    print("Test Accuracy of Network 2 - Dropout = 0.6: " + str(test_accuracy))

    test_accuracy = evaluate_accuracy(test_loader, net2)
    print("Test Accuracy of Network 2: " + str(test_accuracy)) 