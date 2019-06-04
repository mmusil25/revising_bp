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


def getdata_digits():
    from sklearn.datasets import load_digits
    data = load_digits()
    X = data['data']
    y_ = data['target']
    y = np.zeros((len(y_), 10))
    y[np.arange(len(y_)), y_] = 1

    return X, y


def getdata_boston():
    from sklearn.datasets import load_boston
    data = load_boston()
    X = data['data']
    y = data['target']
    y = np.expand_dims(y, -1)

    return X, y


def getdata_diabetes():
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X = data['data']
    y = data['target']
    y = np.expand_dims(y, -1)

    return X, y

def getdata_covtype():
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype()
    X = data['data']
    y_ = data['target'] - 1
    y = np.zeros((len(y_), 7))
    y[np.arange(len(y_)), y_] = 1

    return X, y

def getdata_california_housing():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X = data['data']
    y = data['target']
    y = np.expand_dims(y, -1)

    return X, y

def getdata_olivetti_faces():
    from sklearn.datasets import fetch_olivetti_faces
    data = fetch_olivetti_faces()
    X = data['data']
    y_ = data['target']
    y = np.zeros((len(y_), 40))
    y[np.arange(len(y_)), y_] = 1

    return X, y

def getdata_mnist():
    from sklearn.datasets import fetch_openml
    X, y_ = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X / 255.
    X.astype(dtype=np.float32)
    
    y_ = y_.astype(np.int)
    y = np.zeros((len(y_), 40))
    y[np.arange(len(y_)), y_] = 1
    
    return X, y


def train_eval(model, X_train, y_train, X_test, y_test, epochs, coeff, optimizer, learning_rate, is_shuffle, batch_size):
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

        num_trained = 0
        loc_train_loss = []
        while num_trained < X_train.shape[1]:
            loc_train_loss.append(np.mean(model.train(
                X_train[:, num_trained: num_trained+batch_size], y_train[:, num_trained: num_trained+batch_size], 
                optimizer=optimizer, learning_rate=learning_rate, coeff=coeff)))
            num_trained += batch_size

        train_loss.append(np.mean(loc_train_loss))
        y_test_hat = model.predict(X_test)
        test_loss.append(np.mean(loss_fn.forward(y_test_hat, y_test)))

    return model, train_loss, test_loss


def visualize(curve_dict, args):
    fig = plt.figure()
    for key, val in curve_dict.items():
        plt.plot(np.arange(len(val)), val, label=key)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(f'Dataset: {args.dataset}, Optim: {args.optimizer}, TestRatio:{args.testratio}, \n LR:{args.lr}, HDIMS: {args.hdims}')

    return fig


def id_string(args):
    string = f'{args.optimizer}_T{args.testratio}_EP{args.epochs}_B{args.batch_size}_LR{args.lr}_D_'
    hd_str = ''
    for i in args.hdims:
        hd_str += f'{i}_'
    string += hd_str[:-1]

    return string


datasets = {
    'winequality': getdata_winequality,
    'iris': getdata_iris,
    'digits': getdata_digits,
    'boston': getdata_boston,
    'diabetes': getdata_diabetes,
    'covtype': getdata_covtype,
    'california_housing': getdata_california_housing,
    'olivetti_faces': getdata_olivetti_faces,
    'mnist': getdata_mnist
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
        model, X_train.T, y_train.T, X_test.T, y_test.T, args.epochs, args.coeff, optimizer[args.optimizer], args.lr, args.shuffle, args.batch_size)
    fig = visualize({
        'Train Loss': train_loss,
        'Test Loss': test_loss
    }, args)
    name = id_string(args)
    loss_info = np.vstack((train_loss, test_loss))

    if not os.path.exists(f'./loss_data/{args.dataset}'):
        os.makedirs(f'./loss_data/{args.dataset}')
        np.save(f'./loss_data/{args.dataset}/'+name, loss_info)
    else:
        np.save(f'./loss_data/{args.dataset}/'+name, loss_info)

    if not os.path.exists(f'./loss_figure/{args.dataset}'):
        os.makedirs(f'./loss_figure/{args.dataset}')
        fig.savefig(f'./loss_figure/{args.dataset}/'+name+'.png')
    else:
        fig.savefig(f'./loss_figure/{args.dataset}/'+name+'.png')


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
    parser.add_argument('--layer-coeff', dest='coeff',
                        default=None, nargs='+', type=float)
    parser.add_argument('--batch-size', dest='batch_size', default=50, type=int)

    args = parser.parse_args()
    main(args)
