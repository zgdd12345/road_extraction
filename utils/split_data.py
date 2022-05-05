import os
import random

from utils import is_img


class DataSplit:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_lists = self._get_img_lists()

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, item):
        return self.img_lists[item]

    def _get_img_lists(self):
        img_lists = [img_name for img_name in os.listdir(self.img_path) if is_img(img_name)]
        img_lists = filter(lambda x: x.find('sat') != -1, img_lists)
        return list(map(lambda x: x[:-8], img_lists))

    def _generate_train_lists(self):
        random.shuffle(self.img_lists)
        train_lists = self.img_lists[:int(len(self.img_lists)*0.6)]
        val_lists = self.img_lists[int(len(self.img_lists)*0.6):int(len(self.img_lists)*0.8)]
        test_lists = self.img_lists[int(len(self.img_lists)*0.8):]
        return train_lists, val_lists, test_lists

    def generate_txt(self, save_path):
        """
        write train data 、val data and test data into txt file
        :param save_path:
        :return:without
        """
        train_lists, val_lists, test_lists = self._generate_train_lists()
        # print('len_train:{}, len_val:{}'.format(len(train_lists), len(val_lists)))
        with open(save_path + 'train.txt', 'w') as f:
            for img_num in train_lists:
                img_path = os.path.join(self.img_path, img_num)
                f.write(img_path + '\n')

        with open(save_path + 'validation.txt', 'w') as f:
            for img_num in val_lists:
                img_path = os.path.join(self.img_path, img_num)
                f.write(img_path + '\n')

        with open(save_path + 'test.txt', 'w') as f:
            for img_num in test_lists:
                img_path = os.path.join(self.img_path, img_num)
                f.write(img_path + '\n')


# path = 'F:/data/DeepGlobe Road Extraction Dataset/train/'
path = '/Users/fsm/Road/data/train/'
save_txt = '/Users/fsm/Road/road_extraction/data/'
data_split = DataSplit(path)
data_split.generate_txt(save_txt)

# train_path = './data/train.txt'
# val_path = './data/validation.txt'
#
# train_dataset = RoadDataset(train_path)
# print(len(train_dataset))

# val_dataset = RoadDataset(val_path)
# print(len(val_dataset))

# train_dataset.plot_data(0)
