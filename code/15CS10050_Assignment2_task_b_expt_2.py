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
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2.add(gluon.nn.Dense(num_outputs))


net2_bn = gluon.nn.Sequential()
with net2_bn.name_scope():
    net2_bn.add(gluon.nn.Dense(num_hidden2[0], activation="relu"))
    net2_bn.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net2_bn.add(gluon.nn.Dense(num_hidden2[1], activation="relu"))
    net2_bn.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net2_bn.add(gluon.nn.Dense(num_hidden2[2], activation="relu"))
    net2_bn.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net2_bn.add(gluon.nn.Dense(num_outputs))

net2.load_parameters("../weights/net2.params", ctx=ctx)

if args == '--train':

    train_dataset = CustomDataset('train', val=False)
    train_loader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = CustomDataset('train', val=True)
    val_loader = mx.gluon.data.DataLoader(val_dataset, batch_size, shuffle=False)

    num_examples = len(train_dataset)

    print('Network 2 - With Batch Normalization')

    net2_bn.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net2_bn.collect_params(), 'adam', {'learning_rate': .001})

    epochs = 10
    net2_bn_pts = []
    net2_bn_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            data = data.as_in_context(ctx).reshape((-1, num_inputs))
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net2_bn(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
            
        val_accuracy = evaluate_accuracy(val_loader, net2_bn)
        train_accuracy = evaluate_accuracy(train_loader, net2_bn)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        net2_bn_pts.append(cumulative_loss/num_examples)
        net2_bn_val.append(val_accuracy)

    file_name = "../weights/net2_bn.params"
    net2_bn.save_parameters(file_name)


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
    ax.plot(x, net2_bn_pts, '-b', label='Network2_With_BN')
    ax.plot(x, net2_pts, '-r', label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    leg = ax.legend()
    plt.ylim(top=5)
    fig.show()
    fig.savefig('../plots/loss_task_b_expt_2.png')


    x = range(0, len(net2_val))
    fig, ax = plt.subplots()
    ax.plot(x, net2_bn_val, '-b', label='Network2_With_BN')
    ax.plot(x, net2_val, '-r', label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('val_accuracy')
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/val_task_b_expt_2.png')


if args == '--test':
    
    net2_bn.load_parameters("../weights/net2_bn.params", ctx=ctx)

    net2.load_parameters("../weights/net2.params", ctx=ctx)

    test_dataset = CustomDataset('test', val=False)
    test_loader = mx.gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)
    
    test_accuracy = evaluate_accuracy(test_loader, net2_bn)
    print("Test Accuracy of Network 2 - With Batch Normalization: " + str(test_accuracy))

    test_accuracy = evaluate_accuracy(test_loader, net2)
    print("Test Accuracy of Network 2: " + str(test_accuracy))