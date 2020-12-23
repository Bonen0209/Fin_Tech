import torch
import numpy as np
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def confusion_matrix(output, target):
    try:
        from sklearn.metrics import confusion_matrix as cm
    except ImportError:
        raise RuntimeError("Confusion Matrix requires scikit-learn to be installed.")

    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)

        matrix = cm(target.cpu().numpy(), pred.cpu().numpy())

    fig = plt.figure()
    sns.heatmap(matrix, annot=True, fmt="d")
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    image = torch.from_numpy(buf).permute(2, 0, 1)

    plt.close(fig)

    return image

#TODO
def roc_curve(output, target):
    try:
        from sklearn.metrics import roc_curve as rc
    except ImportError:
        raise RuntimeError("ROC Curve requires scikit-learnto be installed.")

    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        fpr, tpr, _ = rc(target.cpu().numpy(), output[:, 1].cpu().numpy())

    fig = plt.figure()
    plt.plot(fpr, tpr)
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    image = torch.from_numpy(buf).permute(2, 0, 1)

    plt.close(fig)

    return image

#TODO
def precision_recall_curve(output, target):
    try:
        from sklearn.metrics import precision_recall_curve as prc
    except ImportError:
        raise RuntimeError("Precision Recall Curve requires scikit-learn to be installed.")

    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        fpr, tpr, _ = prc(target.cpu().numpy(), output[:, 1].cpu().numpy())

    fig = plt.figure()
    plt.plot(fpr, tpr)
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    image = torch.from_numpy(buf).permute(2, 0, 1)

    plt.close(fig)

    return image
