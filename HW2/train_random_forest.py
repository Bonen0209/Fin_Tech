import argparse
import collections
import numpy as np
import data_loader.data_loaders as module_data
from parse_config import ConfigParser
from trainer import Trainer

from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from scikitplot.metrics import plot_cumulative_gain

# fix random seeds for reproducibility
SEED = 123
np.random.seed(SEED)

def main(config):
    data_loader = config.init_obj('data_loader', module_data)

    X_train, X_test, Y_train, Y_test = train_test_split(data_loader.dataset.datas, data_loader.dataset.targets, test_size=0.2)

    RF = RandomForestClassifier()
    RF.fit(X_train, Y_train)

    preds = RF.predict(X_test)
    preds_prob = RF.predict_proba(X_test)
    print(f'Accuracy score: {accuracy_score(Y_test, preds)}')
    print(f'Precision: {precision_score(Y_test, preds)}')
    print(f'Recall: {recall_score(Y_test, preds)}')
    print(f'F1-score: {f1_score(Y_test, preds)}')

    plot_roc_curve(RF, X_test, Y_test)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig(f'RF_ROC.png')
    plt.close()

    plot_precision_recall_curve(RF, X_test, Y_test)
    plt.legend(loc='lower right')
    plt.savefig(f'RF_PRC.png')
    plt.close()

    preds_prob[:, 0] = 1 - Y_test
    plot_cumulative_gain(Y_test, preds_prob)
    plt.legend(loc='lower right')
    plt.savefig(f'RF_LIFT.png')
    plt.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
