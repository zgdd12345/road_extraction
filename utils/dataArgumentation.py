import numpy as np
import cv2

import random


def random_hue_saturation_value(image, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255),
                                val_shift_limit=(-255, 255), u=0.5):
    """
    色彩饱和度
    :param image:
    :param hue_shift_limit:
    :param sat_shift_limit:
    :param val_shift_limit:
    :param u:
    :return:
    """
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def random_shift_scale_rotate(image, mask, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0), rotate_limit=(-0.0, 0.0),
                              aspect_limit=(-0.0, 0.0), border_mode=cv2.BORDER_CONSTANT, u=0.5):
    """

    :param image:
    :param mask:
    :param shift_limit:
    :param scale_limit:
    :param rotate_limit:
    :param aspect_limit:
    :param border_mode:
    :param u:
    :return:
    """
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=border_mode, borderValue=(0, 0, 0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=border_mode, borderValue=(0, 0, 0,))

    return image, mask


def random_horizontal_flip(image, mask, u=0.5):
    """
    水平翻转
    :param image:
    :param mask:
    :param u:
    :return:
    """
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def random_vertical_flip(image, mask, u=0.5):
    """
    垂直翻转
    :param image:
    :param mask:
    :param u:
    :return:
    """
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def random_rotate_90(image, mask, u=0.5):
    """
    逆时针旋转90deg
    :param image:
    :param mask:
    :param u:
    :return:
    """
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


class ArgumentationDlink:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        img = random_hue_saturation_value(img, u=self.prob)
        img, mask = random_shift_scale_rotate(img, mask, u=self.prob)
        img, mask = random_horizontal_flip(img, mask, u=self.prob)
        img, mask = random_vertical_flip(img, mask, u=self.prob)
        img, mask = random_rotate_90(img, mask, u=self.prob)
        return img, mask


class Rescale:
    """
    随机缩放
    """

    def __init__(self, output_size=1024, prob=0.5):
        self.prob = prob
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, label):
        if random.random() < self.prob:
            raw_h, raw_w = image.shape[:2]

            img = cv2.resize(image, (self.output_size, self.output_size))
            lbl = cv2.resize(label, (self.output_size, self.output_size))

            h, w = img.shape[:2]

            if h > raw_w:
                i = random.randint(0, h - raw_h)
                j = random.randint(0, w - raw_h)
                img = img[i:i + raw_h, j:j + raw_h]
                lbl = lbl[i:i + raw_h, j:j + raw_h]
            else:
                res_h = raw_w - h
                img = cv2.copyMakeBorder(img, res_h, 0, res_h, 0, borderType=cv2.BORDER_REFLECT)
                lbl = cv2.copyMakeBorder(lbl, res_h, 0, res_h, 0, borderType=cv2.BORDER_REFLECT)
            return img, lbl
        else:
            return image, label


class RandomFlip:
    """
    随机翻转
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return img, mask


class RandomRotate90:
    """
    随机旋转n*90
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            # factor = random.randint(0, 4)
            img = np.rot90(img)  # , factor)
            if mask is not None:
                mask = np.rot90(mask)  # , factor)
        return img.copy(), mask.copy()


class Rotate:
    """
    中心旋转
    """

    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class Shift:
    """
    平移
    """

    def __init__(self, limit=50, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            height, width, channel = img.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit + 1, limit + 1, limit + 1, limit + 1,
                                      borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2, :]
            if mask is not None:
                mask1 = cv2.copyMakeBorder(mask, limit + 1, limit + 1, limit + 1, limit + 1,
                                           borderType=cv2.BORDER_REFLECT_101)
                mask = mask1[y1:y2, x1:x2, :]

        return img, mask


class Cutout:
    """
    遮挡问题
    """

    def __init__(self, num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, prob=0.5):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            h = img.shape[0]
            w = img.shape[1]
            # c = img.shape[2]
            # img2 = np.ones([h, w], np.float32)
            for _ in range(self.num_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(max(0, y - self.max_h_size // 2), 0, h)
                y2 = np.clip(max(0, y + self.max_h_size // 2), 0, h)
                x1 = np.clip(max(0, x - self.max_w_size // 2), 0, w)
                x2 = np.clip(max(0, x + self.max_w_size // 2), 0, w)
                img[y1: y2, x1: x2, :] = self.fill_value
                if mask is not None:
                    mask[y1: y2, x1: x2, :] = self.fill_value
        return img, mask


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


data_argumentation = DualCompose([
    # RandomFlip(),
    # RandomRotate90(),
    # Rotate(),
    # Shift(),
    # Rescale()
    ArgumentationDlink()
])
