import sys,os
import numpy as np
path=os.getcwd()
print(path)
sys.path.append(os.path.join(path,'tool'))

from sklearn import metrics
from tool import tool

def accuracy(y_pred,y_true):
    y_true=tool.tensor2numpy(y_true)
    y_pred=tool.tensor2numpy(y_pred)
    return metrics.accuracy_score(y_true=y_true,y_pred=y_pred)

def precision(y_pred,y_true):
    return metrics.precision_score(y_true=tool.tensor2numpy(y_true), y_pred=tool.tensor2numpy(y_pred))

def recall(y_pred,y_true):
    return metrics.recall_score(y_true=tool.tensor2numpy(y_true), y_pred=tool.tensor2numpy(y_pred))

def f1(y_pred,y_true):
    return metrics.f1_score(y_true=tool.tensor2numpy(y_true), y_pred=tool.tensor2numpy(y_pred))

def auc(y_pred,y_true):
    return metrics.f1_score(y_true=tool.tensor2numpy(y_true), y_pred=tool.tensor2numpy(y_pred))






# 请注意，在这个示例中，`predicted_segmentation` 和 `ground_truth` 变量应为代表语义分割图的 NumPy 数组。在实际应用中，您需要根据您的数据结构调整输入和输出。
def evaluate(predicted_segmentation, ground_truth):
    """
    评估语义分割结果，计算召回率、精确率和准确率
    :param predicted_segmentation: 预测的语义分割图，0 表示背景，1 表示前景
    :param ground_truth: 真实的语义分割图，0 表示背景，1 表示前景
    :return: 包含召回率、精确率和准确率的字典
    """
    # 计算 IoU
    intersection = np.logical_and(predicted_segmentation, ground_truth).sum()
    union = np.logical_or(predicted_segmentation, ground_truth).sum()
    iou = intersection / union

    # 计算 TP、FP、FN
    true_positives = np.logical_and(predicted_segmentation, ground_truth).sum()
    false_positives = np.logical_and(np.logical_not(predicted_segmentation), ground_truth).sum()
    false_negatives = np.logical_and(np.logical_not(ground_truth), predicted_segmentation).sum()

    # 计算精确率、召回率和准确率
    accuracy = (true_positives + ground_truth.sum()) / (union + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    precision = true_positives / (true_positives + false_positives + 1e-10)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }

# 示例：评估两个语义分割图
predicted_segmentation = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
ground_truth = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

result = evaluate(predicted_segmentation, ground_truth)
print("Accuracy: {:.2f}%".format(result['accuracy'] * 100))
print("Recall: {:.2f}%".format(result['recall'] * 100))
print("Precision: {:.2f}%".format(result['precision'] * 100))




