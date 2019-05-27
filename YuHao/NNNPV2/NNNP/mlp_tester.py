import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from func.loss_func import SquareError
from mlp import MLP


def getdata_iris():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = data['data']
    y_ = data['target']
    y = np.zeros((len(y_), 3))
    y[np.arange(len(y_)), y_] = 1

    return X, y


def getdata_winequality():
    import pandas as pd
    df = pd.read_csv('./data/winequality-white.csv', sep=';')
    data = df.values

    X = data[:, :-1]
    y = data[:, -1]
    y = np.expand_dims(y, -1)

    return X, y


def train_eval(model, X_train, y_train, X_test, y_test, epochs, optimizer, coeff, learning_rate, is_shuffle):
    train_loss = []
    test_loss = []
    loss_fn = SquareError()
    for _ in range(epochs):
        if is_shuffle:
            x_dim = X_train.shape[0]
            data = np.vstack((X_train, y_train))
            data = data.T
            np.random.shuffle(data)
            X_train = data.T[:x_dim, :]
            y_train = data.T[x_dim:, :]
        train_loss.append(np.mean(model.train(
            X_train, y_train, optimizer=optimizer, learning_rate=learning_rate)))
        y_test_hat = model.predict(X_test)
        test_loss.append(np.mean(loss_fn.forward(y_test_hat, y_test)))

    return model, train_loss, test_loss


def visualize(curve_dict):
    fig = plt.figure()
    for key, val in curve_dict.items():
        plt.plot(np.arange(len(val)), val, label=key)
    plt.xlabel('epoch')
    plt.legend()

    return fig


def id_string(args):
    string = f'{args.dataset}_{args.optimizer}_T{args.testratio}_EP{args.epochs}_LR{args.lr}_D_'
    hd_str = ''
    for i in args.hdims:
        hd_str += f'{i}_'
    string += hd_str[:-1]

    return string


datasets = {
    'winequality': getdata_winequality,
    'iris': getdata_iris
}

optimizer = {
    'sgd': 'sgd',
    'layer': 'sgd_with_layerwise_coeff',
    'neuron': 'sgd_with_elementwise_coeff'
}


def main(args):
    X, y = datasets[args.dataset]()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.testratio, shuffle=args.shuffle)
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]
    model = MLP([in_dim] + args.hdims + [out_dim])
    model, train_loss, test_loss = train_eval(
        model, X_train.T, y_train.T, X_test.T, y_test.T, args.epochs, args.coeff, optimizer[args.optimizer], args.lr, args.shuffle)
    fig = visualize({
        'Train Loss': train_loss,
        'Test Loss': test_loss
    })
    name = id_string(args)
    loss_info = np.vstack((train_loss, test_loss))

    if not os.path.exists('./loss_data'):
        os.makedirs('./loss_data')
        np.save('./loss_data/'+name, loss_info)
    else:
        np.save('./loss_data/'+name, loss_info)

    if not os.path.exists('./loss_figure'):
        os.makedirs('./loss_figure')
        fig.savefig('./loss_figure/'+name+'.png')
    else:
        fig.savefig('./loss_figure/'+name+'.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset',
                        required=True, choices=list(datasets.keys()))
    parser.add_argument('--testratio', dest='testratio',
                        default=0.1, type=float)
    parser.add_argument('--shuffle', dest='shuffle', default=True, type=bool)
    parser.add_argument('--epochs', dest='epochs', default=100, type=int)
    parser.add_argument('--optimizer', dest='optimizer',
                        choices=list(optimizer.keys()))
    parser.add_argument('--lr', dest='lr', default=0.01, type=float)
    parser.add_argument('--hdims', dest='hdims',
                        nargs='+', required=True, type=int)
    parser.add_argument('--layer-coeff', dest='coeff', default=None, nargs='+', type=float)
    
    args = parser.parse_args()
    main(args)
