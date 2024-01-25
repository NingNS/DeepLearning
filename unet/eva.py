import torch
from torch import cosine_similarity
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 获取文件列表
prediction_dir = "final_out_image"
groundtruth_dir = "label111"
prediction_coiled_dir = 'final_out_without_image'
prediction_files = os.listdir(prediction_dir)
groundtruth_files = os.listdir(groundtruth_dir)
prediction_coiled_files = os.listdir(prediction_coiled_dir)

# 初始化真正例、假正例、假负例，通过掩判断真值，通过真值计算
true_positives = 0
false_positives = 0
false_negatives = 0
# 方便进行画图调用sklearn库
predictions = []
labels = []
predictions_coiled = []

for file in prediction_files:
    # 构建预测路径和真实路径
    prediction_path = os.path.join(prediction_dir, file)
    groundtruth_path = os.path.join(groundtruth_dir, file)

    # 加载预测和真实图像
    prediction_image = Image.open(prediction_path)
    groundtruth_image = Image.open(groundtruth_path)

    # 将图像transform一下，再升一维度符合函数维度
    prediction_tensor = transform(prediction_image)
    groundtruth_tensor = transform(groundtruth_image)

    prediction_tensor = prediction_tensor.unsqueeze(0)
    groundtruth_tensor = groundtruth_tensor.unsqueeze(0)

    # 将图像转换为二进制掩码
    prediction_mask = torch.round(prediction_tensor)
    groundtruth_mask = torch.round(groundtruth_tensor)

    # 保存预测和真实标签
    predictions.append(prediction_mask.flatten().numpy())

    labels.append(groundtruth_mask.flatten().numpy())


for file in prediction_coiled_files:
    # 构建预测路径和真实路径
    prediction_coiled_path = os.path.join(prediction_coiled_dir, file)


    # 加载预测和真实图像
    prediction_coiled_image = Image.open(prediction_coiled_path)

    # 将图像transform一下，再升一维度符合函数维度
    prediction_coiled_tensor = transform(prediction_coiled_image)
    prediction_coiled_tensor = prediction_coiled_tensor.unsqueeze(0)

    # 保存预测和真实标签
    predictions_coiled.append(prediction_coiled_tensor.flatten().numpy())


# 计算ROC曲线和AUC值
predictions_coiled = np.concatenate(predictions_coiled)
labels = np.concatenate(labels)
predictions = np.concatenate(predictions)
fpr, tpr, thresholds = metrics.roc_curve(labels, predictions_coiled)
auc = metrics.auc(fpr, tpr)
# 计算PR曲线
precision, recall, _ = metrics.precision_recall_curve(labels, predictions_coiled)
# 计算AUC-PR
auc_pr = metrics.auc(recall, precision)




# 计算混淆矩阵
confusion_matrix = metrics.confusion_matrix(labels, predictions)
confusion_matrix[0][0], confusion_matrix[1][1] = confusion_matrix[1][1], confusion_matrix[0][0]
confusion_matrix[0][1], confusion_matrix[1][0] = confusion_matrix[1][0], confusion_matrix[0][1]
print("Confusion Matrix:")
print(confusion_matrix)

TP = confusion_matrix[0][0]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]
TN = confusion_matrix[1][1]

Accuracy = (TP + TN) / (TP + FP + FN + TN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Specificity = TN / (TN + FP)
F1 = 2 * (Precision * Recall) / (Precision + Recall)

print("Accuracy:", Accuracy)
print("Precision", Precision)
print("Recall", Recall)
print("Specificity", Specificity)
print("F1", F1)
print()


# 绘制PR曲线
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.legend(loc='lower left')
plt.show()

# 绘制ROC曲线
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# 保存曲线图像
plt.show()
print("AUC:", auc)

