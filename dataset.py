from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


mask_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
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

    def __init__(self, file_path):
        self.file_path = file_path
        self.data_list = self._read_txt()

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
                # words = line.split()  # 默认以空格、换行(\n)、制表符(\t)进行分割，大多是"\"
                # images.append((words[0], int(words[1])))  # 存放进imgs列表中
                images.append(line)
        return images

    def plot_data(self, index):
        img_id = self.data_list[index]
        img, mask = self._loader(img_id)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.title('original')
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.title('ground truth')
        plt.imshow(mask, cmap='gray', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

        plt.show()

    @staticmethod
    def _loader(img_id):
        img_name, mask_name = img_id + '_sat.jpg', img_id + '_mask.png'
        img, mask = Image.open(img_name), Image.open(mask_name)
        return img, mask


# train_dataset = RoadDataset(train_path)
# print(len(train_dataset))

# val_dataset = RoadDataset(val_path)
# print(len(val_dataset))

# train_dataset.plot_data(0)
