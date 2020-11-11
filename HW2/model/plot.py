import torch
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def confusion_matrix(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        dim_mat = output.shape[1]
        matrix = torch.zeros((dim_mat, dim_mat))

        for i in range(dim_mat):
            for j in range(dim_mat):
                matrix[i][j] = torch.sum((target == i) * (pred == j)).item()

    fig = plt.figure()
    sns.heatmap(matrix.int().cpu().numpy(), annot=True, fmt="d")
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    image = torch.from_numpy(buf).permute(2, 0, 1)

    plt.close(fig)

    return image

def roc_curve(output, target):
    try:
        from sklearn.metrics import roc_curve as rc
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        fpr, tpr, _ = rc(target.cpu().numpy(), output[:, 1].cpu().numpy())

        print(pred.size(), fpr.shape, tpr.shape)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    image = torch.from_numpy(buf).permute(2, 0, 1)

    plt.close(fig)

    return image

def precision_recall_curve(output, target):
    try:
        from sklearn.metrics import precision_recall_curve as prc
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        fpr, tpr, _ = prc(target.cpu().numpy(), output[:, 1].cpu().numpy())

        print(pred.size(), fpr.shape, tpr.shape)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    image = torch.from_numpy(buf).permute(2, 0, 1)

    plt.close(fig)

    return image

def lift_curve(output, target):
    try:
        from scikitplot.helpers import cumulative_gain_curve
    except ImportError:
        raise RuntimeError("This contrib module requires scikitplot to be installed")

    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)

        percentages, gains1 = cumulative_gain_curve(target.cpu().numpy(), 1- target.cpu().numpy(), 0)
        percentages, gains2 = cumulative_gain_curve(target.cpu().numpy(), output.cpu().numpy()[:, 1], 1)

    fig = plt.figure()
    plt.plot(percentages, gains1)
    plt.plot(percentages, gains2)
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    image = torch.from_numpy(buf).permute(2, 0, 1)

    plt.close(fig)

    return image
