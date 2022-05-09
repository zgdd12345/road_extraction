from PIL import Image
import matplotlib.pyplot as plt
import cv2

from torch.utils.data import Dataset
from torchvision import transforms

from utils.dataArgumentation import data_argumentation, random_hue_saturation_value


mask_transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    #     transforms.Normalize(norm_mean, norm_std),
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class RoadDataset(Dataset):
    """
    加载原始数据集和分割结果
    """

    def __init__(self, file_path, train=True):
        """

        :param file_path: the path of .txt path
        :param train: train val or text
        """
        self.file_path = file_path
        self.data_list = self._read_txt()
        self.train = train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_name = self.data_list[index]
        img, mask = self._loader(img_name)
        return img_transform(img), mask_transform(mask)

    def _read_txt(self):
        images = []  # 定义img的列表
        with open(self.file_path, 'r') as fh:
            for line in fh:
                line = line.rstrip()  # 默认删除的是空白符（'\n', '\r', '\t', ' '）
                images.append(line)
        return images

    def plot_data(self, index):
        img_id = self.data_list[index]
        if not self.train:
            img_name, mask_name = img_id + '_sat.jpg', img_id + '_mask.png'
            img, mask = Image.open(img_name), Image.open(mask_name)
        else:
            img, mask = self._loader(img_id)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.title('original')
        plt.imshow(img.transpose(0, 1, 2))  # mark
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.title('ground truth')
        plt.imshow(mask.transpose(0, 1, 2), cmap='gray', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

        plt.show()

    def _loader(self, img_id):
        img_name, mask_name = img_id + '_sat.jpg', img_id + '_mask.png'
        # img, mask = Image.open(img_name), Image.open(mask_name)
        img, mask = cv2.imread(img_name), cv2.imread(mask_name)  # , cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(os.path.join(root, '{}_sat.jpg').format(id))
        # mask = cv2.imread(os.path.join(root + '{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)

        if self.train:
            img, mask = data_argumentation(img, mask)
            img = random_hue_saturation_value(img)  # 色彩饱和度

        return img, mask  # img_transform(img), mask_transform(mask)

    # @staticmethod
    # def _data_argumentation(img, mask):
    #     img, mask = data_argumentation(img, mask)
    #
    #     # img = random_hue_saturation_value(img, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5),
    #     #                                   val_shift_limit=(-15, 15))
    #     #
    #     # img, mask = random_shift_scale_rotate(img, mask, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1),
    #     #                                       aspect_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
    #     # img, mask = random_horizontal_flip(img, mask)
    #     # img, mask = random_vertical_flip(img, mask)
    #     # img, mask = random_rotate_90(img, mask)
    #     #
    #     # mask = np.expand_dims(mask, axis=2)
    #     # # img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    #     # # mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    #     # mask[mask >= 0.5] = 1
    #     # mask[mask <= 0.5] = 0
    #     return img, mask


# train_path = './data/train.txt'
train_path = '/Users/fsm/Road/road_extraction/data/train.txt'
train_dataset = RoadDataset(train_path, train=True)
print(len(train_dataset))

# val_dataset = RoadDataset(val_path)
# print(len(val_dataset))
for _ in range(10):
    train_dataset.plot_data(1)
