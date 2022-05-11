import numpy as np
import cv2

"""
confusionMetric,真真假假
P\L    P     N

P      TP    FP

N      FN    TN
"""


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def pixel_accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        # acc = (TP) / TP + FP
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc_class = np.nanmean(acc)
        return acc_class

    def mean_intersection_over_union(self):
        miou = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        miou = np.nanmean(miou)
        return miou

    def frequency_weighted_intersection_over_union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
               np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def meansure_pa_miou(num_class, gt_image, pre_image):
    metric = Evaluator(num_class)
    metric.add_batch(gt_image, pre_image)
    acc = metric.pixel_accuracy()
    miou = metric.mean_intersection_over_union()
    print("像素准确度PA:", acc, "平均交互度mIOU:", miou)


if __name__ == '__main__':
    # 求miou，先求混淆矩阵，混淆矩阵的每一行再加上每一列，最后减去对角线上的值；
    # imgPredict = np.array([[0, 0, 1, 0], [1, 1, 0, 2], [2, 2, 1 ,0]])
    # imgLabel = np.array([[0, 0, 0, 1], [1, 1, 2, 2], [2, 2, 0, 0]])
    i = 0
    for i in range(20):
        imgPredict_0 = cv2.imread("datas/test/outputs/20_" + str(i) + "/20_" + str(i) + "_0.png")
        imgPredict_1 = cv2.imread("datas/test/outputs/20_" + str(i) + "/20_" + str(i) + "_1.png")
        imgPredict_2 = cv2.imread("datas/test/outputs/20_" + str(i) + "/20_" + str(i) + "_2.png")
        imgLabel = cv2.imread("datas/mask/20_" + str(i) + ".png")

        # 设置成两类与预测图对应
        imgLabel_0 = imgLabel.copy()
        imgLabel_1 = imgLabel.copy()
        imgLabel_2 = imgLabel.copy()

        imgLabel_0[imgLabel >= 1] = 1  # 背景1和目标0
        height, width, channels = imgLabel_0.shape  # 反转
        for row in range(height):
            for list in range(width):
                for c in range(channels):
                    pv = imgLabel_0[row, list, c]
                    imgLabel_0[row, list, c] = 1 - pv

        imgLabel_1[imgLabel >= 2] = 0  # 把第二类归为背景

        imgLabel_2[imgLabel == 1] = 0  # 把第一类归为背景
        imgLabel_2[imgLabel == 2] = 1

        print(i)

        meansure_pa_miou(2, imgLabel_0, imgPredict_0)
        meansure_pa_miou(2, imgLabel_1, imgPredict_1)
        meansure_pa_miou(2, imgLabel_2, imgPredict_2)
