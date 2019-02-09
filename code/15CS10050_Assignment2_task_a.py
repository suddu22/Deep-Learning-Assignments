import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data import dataset
from data_loader import DataLoader
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import random
import sys

args = sys.argv[1] # to differentiate train and test

ctx = mx.cpu() # use CPU
dl = DataLoader() # data loader that has been provided

batch_size = 64
num_inputs = 784
num_outputs = 10

# Mxnet API dataset
class CustomDataset(dataset.Dataset):
    def __init__(self, mode, val):
        self.X, self.y = dl.load_data(mode, val)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

# function to measure accuracy
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

num_hidden1 = [512, 128, 64, 32, 16]
net1 = gluon.nn.Sequential()
with net1.name_scope():
    net1.add(gluon.nn.Dense(num_hidden1[0], activation="relu"))
    net1.add(gluon.nn.Dense(num_hidden1[1], activation="relu"))
    net1.add(gluon.nn.Dense(num_hidden1[2], activation="relu"))
    net1.add(gluon.nn.Dense(num_hidden1[3], activation="relu"))
    net1.add(gluon.nn.Dense(num_hidden1[4], activation="relu"))
    net1.add(gluon.nn.Dense(num_outputs))

num_hidden2 = [1024, 512, 256]
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2.add(gluon.nn.Dense(num_outputs))

if args == '--train':

    train_dataset = CustomDataset('train', val=False)
    train_loader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True) # create batches using Mxnet's dataloader
    val_dataset = CustomDataset('train', val=True)
    val_loader = mx.gluon.data.DataLoader(val_dataset, batch_size, shuffle=False)

    num_examples = len(train_dataset)

    print('Network 1')

    net1.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net1.collect_params(), 'adam', {'learning_rate': .001})

    epochs = 10
    net1_pts = [] # to store loss values
    net1_val = [] # to store validation accuracy values
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net1(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net1)
        train_accuracy = evaluate_accuracy(train_loader, net1)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net1_pts.append(cumulative_loss/num_examples)
        net1_val.append(val_accuracy)

    file_name = "../weights/net1.params"
    net1.save_parameters(file_name) # save weights to file


    print('Network 2')

    net2.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net2.collect_params(), 'adam', {'learning_rate': .001})

    net2_pts = []
    net2_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net2(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net2)
        train_accuracy = evaluate_accuracy(train_loader, net2)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net2_pts.append(cumulative_loss/num_examples)
        net2_val.append(val_accuracy)

    file_name = "../weights/net2.params"
    net2.save_parameters(file_name)

    # store loss and accuracy values of Network2 for use in other tasks
    with open("../weights/net2_pts.txt", "w") as f:
        for s in net2_pts:
            f.write(str(s) +"\n")

    with open("../weights/net2_val.txt", "w") as f:
        for s in net2_val:
            f.write(str(s) +"\n")

    # training vs epochs plot
    x = range(0, len(net1_pts))
    fig, ax = plt.subplots()
    ax.plot(x, net1_pts, '-b', label='Network1')
    ax.plot(x, net2_pts, '-r', label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(top=4)
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/loss_task_a.png')


    # validation accuracy vs epochs plot
    x = range(0, len(net1_val))
    fig, ax = plt.subplots()
    ax.plot(x, net1_val, '-b', label='Network1')
    ax.plot(x, net2_val, '-r', label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('val_accuracy')
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/val_task_a.png')

if args == '--test':
    
    net1.load_parameters("../weights/net1.params", ctx=ctx) # load weights from file
    net2.load_parameters("../weights/net2.params", ctx=ctx)

    test_dataset = CustomDataset('test', val=False)
    test_loader = mx.gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)

    test_accuracy = evaluate_accuracy(test_loader, net1)
    print("Test Accuracy of Network 1: " + str(test_accuracy))

    test_accuracy = evaluate_accuracy(test_loader, net2)
    print("Test Accuracy of Network 2: " + str(test_accuracy))