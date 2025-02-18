import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir#数据集根目录路径
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))#存储所有图片文件路径的列表，通过 glob.glob 获取所有 .png 格式的文件，并按字母顺序排序。
        self.resize_shape=resize_shape#可选参数，用于指定图片的缩放尺寸

    def __len__(self):
        return len(self.images)#返回数据集中图片的总数

    #transform_image 方法用于加载、处理图像和掩码，使其适合模型训练使用。
    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读取彩色图像
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度掩码
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))  # 创建全零掩码，与图像大小相同
        if self.resize_shape is not None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))  # 缩放图像
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))  # 缩放掩码
        image = image / 255.0  # 归一化图像像素值至 [0, 1]
        mask = mask / 255.0  # 归一化掩码像素值至 [0, 1]
        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)  # 转换图像数据类型为 float32
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)  # 转换掩码数据类型为 float32
        image = np.transpose(image, (2, 0, 1))  # 转置图像维度，变为 (3, H, W)
        mask = np.transpose(mask, (2, 0, 1))  # 转置掩码维度，变为 (1, H, W)
        return image, mask  # 返回处理后的图像和掩码


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]  # 获取指定索引的图片路径
        dir_path, file_name = os.path.split(img_path)  # 分离路径和文件名
        base_dir = os.path.basename(dir_path)  # 获取父目录名

        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)  # 对正常图像进行处理
            has_anomaly = np.array([0], dtype=np.float32)  # 标记为无异常
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')  # 获取掩码文件所在目录
            mask_path = os.path.join(mask_path, base_dir)  # 获取掩码文件所在子目录
            mask_file_name = file_name.split(".")[0] + "_mask.png"  # 构建掩码文件名
            mask_path = os.path.join(mask_path, mask_file_name)  # 构建掩码文件完整路径
            image, mask = self.transform_image(img_path, mask_path)  # 对异常图像及其掩码进行处理
            has_anomaly = np.array([1], dtype=np.float32)  # 标记为有异常

        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx}

        return sample



class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        #self.augmenters：定义了一系列的图像增强操作，包括调整对比度、亮度、锐度、色调饱和度等，使用了 imgaug 库提供的方法。
        # 每个增强器都是 imgaug 库中 iaa 模块的函数，如 iaa.GammaContrast、iaa.MultiplyAndAddToBrightness 等。


        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]
        #使用 imgaug 库中的 Sequential 函数创建一个顺序执行的图像旋转增强器，旋转角度范围为 -90 到 90 度。
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)#返回数据集中的样本数量

    #randAugmenter 方法则随机选择并返回一个包含三个随机增强器的增强器序列，用于数据增强操作
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample
