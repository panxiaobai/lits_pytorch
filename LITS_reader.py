import numpy as np
import os
import glob
import SimpleITK as sitk
import random
import util
from scipy import ndimage

'''
def sitk_read(img_path):
    nda = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(nda) #(155,240,240)
    zero = np.zeros([5, 240, 240])
    nda = np.concatenate([zero, nda], axis=0) #(160,240,240)
    nda = nda.transpose(1, 2, 0) #(240,240,160)
    return nda
'''
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def norm_img(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def sitk_read_row(img_path, resize_scale=1):
    nda = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(nda)  # channel first
    nda=ndimage.zoom(nda,[resize_scale,resize_scale,resize_scale],order=0)

    return nda


def make_one_hot_3d(x, n):
    one_hot = np.zeros([x.shape[0], x.shape[1], x.shape[2], n])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for v in range(x.shape[2]):
                one_hot[i, j, v, int(x[i, j, v])] = 1

    return one_hot


class LITS_reader:
    def __init__(self, data_init=False, data_fix=False):
        self.row_root_path = '/root/userfolder/PY/data/LITS/'
        self.data_root_path = '/root/userfolder/PY/data/LITS/fixed/'
        if data_fix:
            self.fix_data()
        if data_init:
            self.init_data()

        self.train_name_list = self.load_file_name_list(self.data_root_path + "train_name_list.txt")
        self.val_name_list = self.load_file_name_list(self.data_root_path + "val_name_list.txt")
        self.test_name_list = self.load_file_name_list(self.data_root_path + "test_name_list.txt")

        self.n_train_file = len(self.train_name_list)
        self.n_val_file = len(self.val_name_list)
        self.n_test_file = len(self.test_name_list)

        self.train_batch_index = 0
        self.val_batch_index = 0
        self.test_batch_index = 0

        self.n_labels = 3

    def write_train_val_test_name_list(self):
        data_name_list = np.zeros([131], dtype='int32')
        for i in range(131):
            data_name_list[i] = i
        # data_name_list = os.listdir(self.data_root_path + "/")
        random.shuffle(data_name_list)
        length = len(data_name_list)
        n_train_file = int(length / 10 * 8)
        n_val_file = int(length / 10 * 1)
        train_name_list = data_name_list[0:n_train_file]
        val_name_list = data_name_list[n_train_file:(n_train_file + n_val_file)]
        test_name_list = data_name_list[(n_train_file + n_val_file):len(data_name_list)]
        self.write_name_list(train_name_list, "train_name_list.txt")
        self.write_name_list(val_name_list, "val_name_list.txt")
        self.write_name_list(test_name_list, "test_name_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(self.data_root_path + file_name, 'w')
        for i in range(len(name_list)):
            f.write(str(name_list[i]) + "\n")
        f.close()

    def init_data(self):
        self.write_train_val_test_name_list()

    def fix_data(self):
        upper = 200
        lower = -200
        expand_slice = 20  # 轴向上向外扩张的slice数量
        size = 48  # 取样的slice数量
        stride = 3  # 取样的步长
        down_scale = 0.5
        slice_thickness = 2

        for ct_file in os.listdir(self.row_root_path + 'data/'):
            print(ct_file)
            # 将CT和金标准入读内存
            ct = sitk.ReadImage(os.path.join(self.row_root_path + 'data/', ct_file), sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)

            seg = sitk.ReadImage(os.path.join(self.row_root_path + 'label/', ct_file.replace('volume', 'segmentation')),
                                 sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            print(ct_array.shape, seg_array.shape)

            # 将金标准中肝脏和肝肿瘤的标签融合为一个
            seg_array[seg_array > 0] = 1

            # 将灰度值在阈值之外的截断掉
            ct_array[ct_array > upper] = upper
            ct_array[ct_array < lower] = lower

            # 找到肝脏区域开始和结束的slice，并各向外扩张
            z = np.any(seg_array, axis=(1, 2))
            start_slice, end_slice = np.where(z)[0][[0, -1]]

            # 两个方向上各扩张个slice
            if start_slice - expand_slice < 0:
                start_slice = 0
            else:
                start_slice -= expand_slice

            if end_slice + expand_slice >= seg_array.shape[0]:
                end_slice = seg_array.shape[0] - 1
            else:
                end_slice += expand_slice

            print(str(start_slice) + '--' + str(end_slice))
            # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
            if end_slice - start_slice + 1 < size:
                print('!!!!!!!!!!!!!!!!')
                print(ct_file, 'too little slice')
                print('!!!!!!!!!!!!!!!!')
                continue

            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array = seg_array[start_slice:end_slice + 1, :, :]

            new_ct = sitk.GetImageFromArray(ct_array)
            new_seg = sitk.GetImageFromArray(seg_array)

            sitk.WriteImage(new_ct, os.path.join(self.data_root_path + 'data/', ct_file))
            sitk.WriteImage(new_seg,
                            os.path.join(self.data_root_path + 'label/', ct_file.replace('volume', 'segmentation')))

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                    pass
                file_name_list.append(lines)
                pass
        return file_name_list

    def get_np_data_3d(self, data_name, resize_scale=1):
        data_np = sitk_read_row(self.data_root_path + 'data/' + 'volume-' + data_name + '.nii',
                                resize_scale=resize_scale)
        data_np=norm_img(data_np)
        label_np = sitk_read_row(self.data_root_path + 'label/' + 'segmentation-' + data_name + '.nii',
                                 resize_scale=resize_scale)

        return data_np, label_np

    def next_train_batch_3d_sub_by_index(self, train_batch_size, crop_size, index,resize_scale=1):
        train_imgs = np.zeros([train_batch_size, crop_size[0], crop_size[1], crop_size[2], 1])
        train_labels = np.zeros([train_batch_size, crop_size[0], crop_size[1], crop_size[2], self.n_labels])
        img, label = self.get_np_data_3d(self.train_name_list[index],resize_scale=resize_scale)
        for i in range(train_batch_size):
            sub_img, sub_label = util.random_crop_3d(img, label, crop_size)

            sub_img = sub_img[:, :, :, np.newaxis]
            sub_label_onehot = make_one_hot_3d(sub_label, self.n_labels)

            train_imgs[i] = sub_img
            train_labels[i] = sub_label_onehot

        return train_imgs, train_labels

    def next_train_batch_3d_sub(self, train_batch_size, crop_size):
        self.n_train_steps_per_epoch = self.n_train_file // 1
        train_imgs = np.zeros([train_batch_size, crop_size[0], crop_size[1], crop_size[2], 1])
        train_labels = np.zeros([train_batch_size, crop_size[0], crop_size[1], crop_size[2], self.n_labels])
        if self.train_batch_index >= self.n_train_steps_per_epoch:
            self.train_batch_index = 0
        img, label = self.get_np_data_3d(self.train_name_list[self.train_batch_index])
        for i in range(train_batch_size):
            sub_img, sub_label = util.random_crop_3d(img, label, crop_size)
            '''
            num=0
            num_0=0
            num_1=0
            num_2=0
            for z in range(sub_label.shape[0]):
                for x in range(sub_label.shape[1]):
                    for c in range(sub_label.shape[2]):
                        if sub_label[z][x][c]!=0:
                            num+=1
                        if sub_label[z][x][c]==0:
                            num_0+=1
                        if sub_label[z][x][c]==1:
                            num_1+=1
                        if sub_label[z][x][c]==2:
                            num_2+=1
            print('-----')
            print(num)
            print(num_0)
            print(num_1)
            print(num_2)
            print('-----')
            '''
            sub_img = sub_img[:, :, :, np.newaxis]
            sub_label_onehot = make_one_hot_3d(sub_label, self.n_labels)
            '''
            num = 0
            num_0 = 0
            num_1 = 0
            num_2 = 0
            for z in range(sub_label.shape[0]):
                for x in range(sub_label.shape[1]):
                    for c in range(sub_label.shape[2]):
                        if sub_label_onehot[z][x][c][0] == 1:
                            num_0 += 1
                        if sub_label_onehot[z][x][c][1] == 1:
                            num_1 += 1
                        if sub_label_onehot[z][x][c][2] == 1:
                            num_2 += 1
            print('-----')
            print(num)
            print(num_0)
            print(num_1)
            print(num_2)
            print('-----')
            '''
            train_imgs[i] = sub_img
            train_labels[i] = sub_label_onehot

        self.train_batch_index += 1
        return train_imgs, train_labels

    def next_val_batch_3d_sub_by_index(self, val_batch_size, crop_size, index,resize_scale=1):
        val_imgs = np.zeros([val_batch_size, crop_size[0], crop_size[1], crop_size[2], 1])
        val_labels = np.zeros([val_batch_size, crop_size[0], crop_size[1], crop_size[2], self.n_labels])
        img, label = self.get_np_data_3d(self.val_name_list[index],resize_scale=resize_scale)
        for i in range(val_batch_size):
            sub_img, sub_label = util.random_crop_3d(img, label, crop_size)

            sub_img = sub_img[:, :, :, np.newaxis]
            sub_label_onehot = make_one_hot_3d(sub_label, self.n_labels)

            val_imgs[i] = sub_img
            val_labels[i] = sub_label_onehot

        return val_imgs, val_labels

    def next_val_batch_3d_sub(self, val_batch_size, crop_size):
        self.n_val_steps_per_epoch = self.n_val_file // 1
        val_imgs = np.zeros([val_batch_size, crop_size[0], crop_size[1], crop_size[2], 1])
        val_labels = np.zeros([val_batch_size, crop_size[0], crop_size[1], crop_size[2], self.n_labels])
        if self.val_batch_index >= self.n_val_steps_per_epoch:
            self.val_batch_index = 0
        img, label = self.get_np_data_3d(self.val_name_list[self.val_batch_index])
        for i in range(val_batch_size):
            sub_img, sub_label = util.random_crop_3d(img, label, crop_size)

            sub_img = sub_img[:, :, :, np.newaxis]
            sub_label_onehot = make_one_hot_3d(sub_label, self.n_labels)

            val_imgs[i] = sub_img
            val_labels[i] = sub_label_onehot

        self.val_batch_index += 1
        return val_imgs, val_labels

    def next_test_img(self):
        self.n_test_steps_per_epoch = self.n_test_file // 1
        if self.test_batch_index >= self.n_test_steps_per_epoch:
            self.test_batch_index = 0
        img, label = self.get_np_data_3d(self.test_name_list[self.test_batch_index], resize_scale=0.5)

        img = img[np.newaxis, :, :, :, np.newaxis]
        label = make_one_hot_3d(label, self.n_labels)
        label = label[np.newaxis, :]

        self.test_batch_index += 1

        return img, label

    def next_train_batch_3d(self, train_batch_size):
        return None


def main():
    reader = LITS_reader(data_fix=False)
    img, label = reader.next_val_batch_3d_sub(8, [32, 64, 64])
    print(img.shape)
    print(label.shape)


if __name__ == '__main__':
    main()
