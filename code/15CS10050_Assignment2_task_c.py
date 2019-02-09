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

net2.load_parameters("../weights/net2.params", ctx=ctx)

# Logistic regression classifiers
lr_hl1 = gluon.nn.Dense(num_outputs)
lr_hl2 = gluon.nn.Dense(num_outputs)
lr_hl3 = gluon.nn.Dense(num_outputs)

if args == '--train':

    train_dataset = CustomDataset('train', val=False)
    train_loader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = CustomDataset('train', val=True)
    val_loader = mx.gluon.data.DataLoader(val_dataset, batch_size, shuffle=False)

    num_examples = len(train_dataset)

    full_train_dataset = mx.ndarray.cast(mx.nd.array(train_loader._dataset.X), dtype='float32')
    full_val_dataset = mx.ndarray.cast(mx.nd.array(val_loader._dataset.X), dtype='float32')

    hl1_out_train = net2._children['0'].forward(full_train_dataset) # output of hidden layer 1
    hl2_out_train = net2._children['1'].forward(hl1_out_train) # output of hidden layer 2
    hl3_out_train = net2._children['2'].forward(hl2_out_train) # output of hidden layer 3

    hl1_out_val = net2._children['0'].forward(full_val_dataset)
    hl2_out_val = net2._children['1'].forward(hl1_out_val)
    hl3_out_val = net2._children['2'].forward(hl2_out_val)

    print('Logistic Regression - Using Output of Hidden Layer 1')

    lr_hl1.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(lr_hl1.collect_params(), 'adam', {'learning_rate': .0003})


    train_loader._dataset.X = hl1_out_train
    val_loader._dataset.X = hl1_out_val


    epochs = 15
    lr_hl1_pts = []
    lr_hl1_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = lr_hl1(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()

        val_accuracy = evaluate_accuracy(val_loader, lr_hl1)
        train_accuracy = evaluate_accuracy(train_loader, lr_hl1)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        lr_hl1_pts.append(cumulative_loss/num_examples)
        lr_hl1_val.append(val_accuracy)

    file_name = "../weights/lr_hl1.params"
    lr_hl1.save_parameters(file_name)


    print('Logistic Regression - Using Output of Hidden Layer 2')


    lr_hl2.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(lr_hl2.collect_params(), 'adam', {'learning_rate': .0003})

    train_loader._dataset.X = hl2_out_train
    val_loader._dataset.X = hl2_out_val

    lr_hl2_pts = []
    lr_hl2_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = lr_hl2(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()

        val_accuracy = evaluate_accuracy(val_loader, lr_hl2)
        train_accuracy = evaluate_accuracy(train_loader, lr_hl2)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        lr_hl2_pts.append(cumulative_loss/num_examples)
        lr_hl2_val.append(val_accuracy)

    file_name = "../weights/lr_hl2.params"
    lr_hl2.save_parameters(file_name)


    print('Logistic Regression - Using Output of Hidden Layer 3')


    lr_hl3.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(lr_hl3.collect_params(), 'adam', {'learning_rate': .0003})

    train_loader._dataset.X = hl3_out_train
    val_loader._dataset.X = hl3_out_val

    lr_hl3_pts = []
    lr_hl3_val = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = mx.ndarray.cast(data, dtype='float32')
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = lr_hl3(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()

        val_accuracy = evaluate_accuracy(val_loader, lr_hl3)
        train_accuracy = evaluate_accuracy(train_loader, lr_hl3)
        print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" %
            (e, cumulative_loss/num_examples, train_accuracy, val_accuracy))
        lr_hl3_pts.append(cumulative_loss/num_examples)
        lr_hl3_val.append(val_accuracy)

    file_name = "../weights/lr_hl3.params"
    lr_hl3.save_parameters(file_name)


    net2_pts = []
    net2_val = []

    with open("../weights/net2_pts.txt", "r") as f:
        for line in f:
            net2_pts.append(float(line.strip()))

    with open("../weights/net2_val.txt", "r") as f:
        for line in f:
            net2_val.append(float(line.strip()))


    x = range(0, len(lr_hl1_pts))
    fig, ax = plt.subplots()
    ax.plot(x, lr_hl1_pts, label='Log. Reg. using HL1')
    ax.plot(x, lr_hl2_pts, label='Log. Reg. using HL2')
    ax.plot(x, lr_hl3_pts, label='Log. Reg. using HL3')
    # ax.plot(x, net2_pts, label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(top=5)
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/loss_task_c.png')

    x = range(0, len(lr_hl1_val))
    fig, ax = plt.subplots()
    ax.plot(x, lr_hl1_val, label='Log. Reg. using HL1')
    ax.plot(x, lr_hl2_val, label='Log. Reg. using HL2')
    ax.plot(x, lr_hl3_val, label='Log. Reg. using HL3')
    # ax.plot(x, net2_val, label='Network2')
    plt.xlabel('epoch')
    plt.ylabel('val_accuracy')
    leg = ax.legend()
    fig.show()
    fig.savefig('../plots/val_task_c.png')




if args == '--test':
    
    lr_hl1.load_parameters("../weights/lr_hl1.params", ctx=ctx)
    lr_hl2.load_parameters("../weights/lr_hl2.params", ctx=ctx)
    lr_hl3.load_parameters("../weights/lr_hl3.params", ctx=ctx)
    net2.load_parameters("../weights/net2.params", ctx=ctx)

    test_dataset = CustomDataset('test', val=False)
    test_loader = mx.gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)

    full_test_dataset = mx.ndarray.cast(mx.nd.array(test_loader._dataset.X), dtype='float32')

    hl1_out_test = net2._children['0'].forward(full_test_dataset)
    hl2_out_test = net2._children['1'].forward(hl1_out_test)
    hl3_out_test = net2._children['2'].forward(hl2_out_test)
    
    test_loader._dataset.X = hl1_out_test
    test_accuracy = evaluate_accuracy(test_loader, lr_hl1)
    print("Test Accuracy of Logistic Regression Classifier - Using Output of Hidden Layer 1: " + str(test_accuracy))

    test_loader._dataset.X = hl2_out_test
    test_accuracy = evaluate_accuracy(test_loader, lr_hl2)
    print("Test Accuracy of Logistic Regression Classifier - Using Output of Hidden Layer 2: " + str(test_accuracy))

    test_loader._dataset.X = hl3_out_test
    test_accuracy = evaluate_accuracy(test_loader, lr_hl3)
    print("Test Accuracy of Logistic Regression Classifier - Using Output of Hidden Layer 3: " + str(test_accuracy))

    # test_accuracy = evaluate_accuracy(test_loader, net2)
    # print("Test Accuracy of Network 2: " + str(test_accuracy))