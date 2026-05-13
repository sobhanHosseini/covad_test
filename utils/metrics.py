import numpy as np
from sklearn.metrics import precision_recall_curve


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def binary_accuracy(output, target):
    pred = output.cpu() >= 0.5
    target = target.cpu()
    accuracy = (pred.int().eq(target.int())).sum()
    accuracy = accuracy * 100 / np.prod(np.array(target.size()))
    return accuracy


def compute_image_f1(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    best_idx = np.argmax(f1)
    return f1[best_idx], thresholds[best_idx]


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)
