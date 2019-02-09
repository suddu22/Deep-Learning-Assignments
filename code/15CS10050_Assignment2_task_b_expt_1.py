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
net2_normal = gluon.nn.Sequential()
with net2_normal.name_scope():
    net2_normal.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2_normal.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2_normal.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2_normal.add(gluon.nn.Dense(num_outputs))

net2_xavier = gluon.nn.Sequential()
with net2_xavier.name_scope():
    net2_xavier.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2_xavier.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2_xavier.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2_xavier.add(gluon.nn.Dense(num_outputs))

net2_orthogonal = gluon.nn.Sequential()
with net2_orthogonal.name_scope():
    net2_orthogonal.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2_orthogonal.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2_orthogonal.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2_orthogonal.add(gluon.nn.Dense(num_outputs))

num_hidden2 = [1024, 512, 256]
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2.add(gluon.nn.Dense(num_outputs))

net2.load_parameters("../weights/net2.params", ctx=ctx) # load the stored weights of Network2 so that we do not have to retrain it

if args == '--train':

    train_dataset = CustomDataset('train', val=False)
    train_loader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = CustomDataset('train', val=True)
    val_loader = mx.gluon.data.DataLoader(val_dataset, batch_size, shuffle=False)

    num_examples = len(train_dataset)

    print('Network 2 - Normal Initialization')

    net2_normal.collect_params().initialize(mx.init.Normal(sigma=0.05), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net2_normal.collect_params(), 'adam', {'learning_rate': .001})

    epochs = 10
    net2_normal_pts = []
    net2_normal_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net2_normal(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()

        val_accuracy = evaluate_accuracy(val_loader, net2_normal)
        train_accuracy = evaluate_accuracy(train_loader, net2_normal)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net2_normal_pts.append(cumulative_loss/num_examples)
        net2_normal_val.append(val_accuracy)

    file_name = "../weights/net2_normal.params"
    net2_normal.save_parameters(file_name)


    print('Network 2 - Xavier Initialization')


    net2_xavier.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net2_xavier.collect_params(), 'adam', {'learning_rate': .001})

    net2_xavier_pts = []
    net2_xavier_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net2_xavier(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()

        val_accuracy = evaluate_accuracy(val_loader, net2_xavier)
        train_accuracy = evaluate_accuracy(train_loader, net2_xavier)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net2_xavier_pts.append(cumulative_loss/num_examples)
        net2_xavier_val.append(val_accuracy)

    file_name = "../weights/net2_xavier.params"
    net2_xavier.save_parameters(file_name)


    print('Network 2 - Orthogonal Initialization')


    net2_orthogonal.collect_params().initialize(mx.init.Orthogonal(), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net2_orthogonal.collect_params(), 'adam', {'learning_rate': .001})

    net2_orthogonal_pts = []
    net2_orthogonal_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net2_orthogonal(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()

        val_accuracy = evaluate_accuracy(val_loader, net2_orthogonal)
        train_accuracy = evaluate_accuracy(train_loader, net2_orthogonal)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net2_orthogonal_pts.append(cumulative_loss/num_examples)
        net2_orthogonal_val.append(val_accuracy)

    file_name = "../weights/net2_orthogonal.params"
    net2_orthogonal.save_parameters(file_name)


    net2_pts = []
    net2_val = []

    # read the stored loss and accuracy values of Network2
    with open("../weights/net2_pts.txt", "r") as f:
        for line in f:
            net2_pts.append(float(line.strip()))

    with open("../weights/net2_val.txt", "r") as f:
        for line in f:
            net2_val.append(float(line.strip()))


    x = range(0, len(net2_pts))
    fig, ax = plt.subplots()
    ax.plot(x, net2_normal_pts, label='Network2_Normal')
    ax.plot(x, net2_xavier_pts, label='Network2_Xavier')
    ax.plot(x, net2_orthogonal_pts, label='Network2_Orthogonal')
    ax.plot(x, net2_pts, label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(top=5)
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/loss_task_b_expt_1.png')

    x = range(0, len(net2_val))
    fig, ax = plt.subplots()
    ax.plot(x, net2_normal_val, label='Network2_Normal')
    ax.plot(x, net2_xavier_val, label='Network2_Xavier')
    ax.plot(x, net2_orthogonal_val, label='Network2_Orthogonal')
    ax.plot(x, net2_val, label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('val_accuracy')
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/val_task_b_expt_1.png')




if args == '--test':
    
    net2_normal.load_parameters("../weights/net2_normal.params", ctx=ctx)
    net2_xavier.load_parameters("../weights/net2_xavier.params", ctx=ctx)
    net2_orthogonal.load_parameters("../weights/net2_orthogonal.params", ctx=ctx)
    net2.load_parameters("../weights/net2.params", ctx=ctx)

    test_dataset = CustomDataset('test', val=False)
    test_loader = mx.gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)
    
    test_accuracy = evaluate_accuracy(test_loader, net2_normal)
    print("Test Accuracy of Network 2 - Normal Initialization: " + str(test_accuracy))

    test_accuracy = evaluate_accuracy(test_loader, net2_xavier)
    print("Test Accuracy of Network 2 - Xavier Initialization: " + str(test_accuracy))

    test_accuracy = evaluate_accuracy(test_loader, net2_orthogonal)
    print("Test Accuracy of Network 2 - Orthogonal Initialization: " + str(test_accuracy))

    test_accuracy = evaluate_accuracy(test_loader, net2)
    print("Test Accuracy of Network 2: " + str(test_accuracy)) 