import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from yolov3.utils.utils import xyxy2xywh
from yolov3.utils.utils import get_background
from utils.utils_funs import get_rotation,getFiles_any,data_augment_one_myycb_syn

class LoadImages:  # for inference
    def __init__(self, path, img_size=416):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.height = img_size

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'File Not Found ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0


        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, img_size=416):
        self.cam = cv2.VideoCapture(0)
        self.height = img_size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img_path, img, img0

    def __len__(self):
        return 0


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=[416,416], augment=True):
        with open(path, 'r') as file:
            self.img_files = file.read().splitlines()
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))
        assert len(self.img_files) > 0, 'No images found in %s' % path
        self.img_size = img_size
        self.augment = augment
        self.label_files = [
            x.replace('rgb_','').replace('images', 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')
            for x in self.img_files]
        self.seg_files = [
            x.replace('img/rgb_', 'seg/seg_').replace('images', 'labels').replace('.bmp', '.png').replace('.jpg', '.png').replace('.png', '.png')
            for x in self.img_files]
        self.bg0  = '/home/wei/Documents/code/pipe_line/COCO/coco/images/trainval35k/'
        self.bg = getFiles_any(self.bg0, '.jpg')

    def __len__(self):
        return len(self.seg_files)

    def __getitem__(self, index):

        img_path = self.img_files[index]
        label_path = self.label_files[index]
        seg_path = self.seg_files[index]

        img = cv2.imread(img_path)  # BGR
        seg = cv2.imread(seg_path)[:,:,0]/255.0 # BGR
        # seg = cv2.resize(seg, (416,416), interpolation=cv2.INTER_AREA)

        idx = np.random.choice(len(self.bg), 1, replace=False)
        bimg = cv2.imread(self.bg0 + self.bg[idx[0]])
        bimg = cv2.resize(bimg, (img.shape[1],img.shape[0]))

        img = get_background(img, seg, bimg)

        assert img is not None, 'File Not Found ' + img_path

        # ll=len(label_path)
        # print(label_path)
        # assert ll>0

        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50  # must be < 1.0
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, None, 255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, None, 255, out=V)

            img_hsv[:, :, 1] = S  # .astype(np.uint8)
            img_hsv[:, :, 2] = V  # .astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=self.img_size)


        # print('ratio: ', ratio)
        # Load labels
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as file:
                lines = file.read().splitlines()

            x = np.array([x.split() for x in lines], dtype=np.float32)
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                x[:, 1]=max(x[:, 1],0)
                x[:, 2]=max(x[:, 2],0)
                # print()
                # print(labels, padw,padh)

                padw=0
                padh=0
                ratio1=self.img_size[1]/w
                ratio2 = self.img_size[0] / h
                labels[:, 1] = ratio1 * w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = ratio2 * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 1] = max(labels[:, 1], 0)
                labels[:, 2] = max(labels[:, 2], 0)

                labels[:, 3] = ratio1 * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = ratio2 * h * (x[:, 2] + x[:, 4] / 2) + padh

                labels[:, 3] = min(labels[:, 3], self.img_size[1])
                labels[:, 4] = min(labels[:, 4], self.img_size[0])
                # print('labb: ',labels)
                # tess

        # Augment image and labels
        if self.augment:
            img, labels = random_affine(img, labels, degrees=(-10, 10), translate=(0.15, 0.15), scale=(0.80, 1.20))

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            a = xyxy2xywh(labels[:, 1:5]) ###[[     219.92      128.63      35.722      38.799]]
            labels[:, [1,3]] = a[0,[0,2]] / self.img_size[1] ### w
            labels[:, [2,4]] = a[0,[1,3]] / self.img_size[0] ### h

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() > 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            # print()
            # print(i)
            # print(l)
            if len(l)==0:
                continue
            l[:, 0] = i  # add target image index for build_targets()
            # print(l)
        return torch.stack(img, 0), torch.cat(label, 0), path, hw


class LoadImagesAndLabels_syn(Dataset):  # for training/testing
    def __init__(self, path, model,K,img_size=[416, 416], augment=True, syn=False):
        with open(path, 'r') as file:
            self.img_files = file.read().splitlines()
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))
        assert len(self.img_files) > 0, 'No images found in %s' % path
        self.img_size = img_size
        self.augment = augment
        self.label_files = [
            x.replace('rgb_', '').replace('images', 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace(
                '.png', '.txt')
            for x in self.img_files]
        self.seg_files = [
            x.replace('img/rgb_', 'seg/seg_').replace('images', 'labels').replace('.bmp', '.png').replace('.jpg',
                                                                                                          '.png').replace(
                '.png', '.png')
            for x in self.img_files]
        self.bg0 = '/home/wei/Documents/code/pipe_line/COCO/coco/images/trainval35k/'
        self.bg = getFiles_any(self.bg0, '.jpg')
        self.syn = syn
        self.model = model
        self.K = K
    def __len__(self):
        return len(self.seg_files)

    def __getitem__(self, index):

        if self.syn:
            choose = np.random.random()
            if choose > 0.5:
                img_path = self.img_files[index]
                label_path = self.label_files[index]
                seg_path = self.seg_files[index]

                img = cv2.imread(img_path)  # BGR
                seg = cv2.imread(seg_path)[:, :, 0] / 255.0  # BGR
                if os.path.isfile(label_path):
                    with open(label_path, 'r') as file:
                        lines = file.read().splitlines()

                    label = np.array([x.split() for x in lines], dtype=np.float32)
            else:
                label = np.zeros((4),dtype=np.float32)
                x = np.random.randint(0,360)
                y = np.random.randint(0,180)
                z = np.random.randint(0,360)

                xt = np.random.randint(-360, 360)
                yt = np.random.randint(-180, 180)
                zt = np.random.randint(500, 1500)
                t = np.array([xt,yt,zt])
                R = get_rotation(x,y,z)

                img = data_augment_one_myycb_syn(R, t, self.K, self.model, [640,480], mode='rgb')

                H = img.shape[0]
                W = img.shape[1]
                r,c  = np.where(img[0]>0)
                enl = 10
                xmin = max(c.min() - enl, 0)
                ymin = max(r.min() - enl, 0)
                wid = min(c.max() + enl, W) - c.min()
                hig = min(r.max() + enl, H) - r.min()


                # cv2.rectangle(seg, (xmin,ymin), (xmin+wid,ymin+hig), (255, 0, 0), 3)
                # cv2.imshow('test',seg)
                # cv2.waitKey()

                b1 = (xmin + wid / 2) / W
                b2 = (ymin + hig / 2) / H
                b3 = wid / W
                b4 = hig / H
                label[0]  = b1
                label[1] = b2
                label[2] = b3
                label[3] = b4

                seg = img>0
            # seg = cv2.resize(seg, (416,416), interpolation=cv2.INTER_AREA)

            idx = np.random.choice(len(self.bg), 1, replace=False)
            bimg = cv2.imread(self.bg0 + self.bg[idx[0]])
            bimg = cv2.resize(bimg, (img.shape[1], img.shape[0]))

            img = get_background(img, seg, bimg)

            assert img is not None, 'File Not Found ' + img_path

            # ll=len(label_path)
            # print(label_path)
            # assert ll>0

            augment_hsv = True
            if self.augment and augment_hsv:
                # SV augmentation by 50%
                fraction = 0.50  # must be < 1.0
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, None, 255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, None, 255, out=V)

                img_hsv[:, :, 1] = S  # .astype(np.uint8)
                img_hsv[:, :, 2] = V  # .astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            h, w, _ = img.shape
            img, ratio, padw, padh = letterbox(img, height=self.img_size)

            # print('ratio: ', ratio)
            # Load labels
            labels = []
            x = label
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                x[:, 1] = max(x[:, 1], 0)
                x[:, 2] = max(x[:, 2], 0)
                # print()
                # print(labels, padw,padh)

                padw = 0
                padh = 0
                ratio1 = self.img_size[1] / w
                ratio2 = self.img_size[0] / h
                labels[:, 1] = ratio1 * w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = ratio2 * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 1] = max(labels[:, 1], 0)
                labels[:, 2] = max(labels[:, 2], 0)

                labels[:, 3] = ratio1 * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = ratio2 * h * (x[:, 2] + x[:, 4] / 2) + padh

                labels[:, 3] = min(labels[:, 3], self.img_size[1])
                labels[:, 4] = min(labels[:, 4], self.img_size[0])
                # print('labb: ',labels)
                # tess

            # Augment image and labels
            if self.augment:
                img, labels = random_affine(img, labels, degrees=(-10, 10), translate=(0.15, 0.15), scale=(0.80, 1.20))

            nL = len(labels)  # number of labels
            if nL:
                # convert xyxy to xywh
                a = xyxy2xywh(labels[:, 1:5])  ###[[     219.92      128.63      35.722      38.799]]
                labels[:, [1, 3]] = a[0, [0, 2]] / self.img_size[1]  ### w
                labels[:, [2, 4]] = a[0, [1, 3]] / self.img_size[0]  ### h

            if self.augment:
                # random left-right flip
                lr_flip = True
                if lr_flip and random.random() > 0.5:
                    img = np.fliplr(img)
                    if nL:
                        labels[:, 1] = 1 - labels[:, 1]

                # random up-down flip
                ud_flip = False
                if ud_flip and random.random() > 0.5:
                    img = np.flipud(img)
                    if nL:
                        labels[:, 2] = 1 - labels[:, 2]

            labels_out = torch.zeros((nL, 6))
            if nL:
                labels_out[:, 1:] = torch.from_numpy(labels)

            # Normalize
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            # print()
            # print(i)
            # print(l)
            if len(l) == 0:
                continue
            l[:, 0] = i  # add target image index for build_targets()
            # print(l)
        return torch.stack(img, 0), torch.cat(label, 0), path, hw

def letterbox(img, height, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = max(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = max(0,(height[1] - new_shape[0]) / 2)  # width padding
    dh = max((height[0] - new_shape[1]) / 2,0) # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, max(0,top), max(0,bottom), max(0,left), max(0,right), cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def random_affine(img, targets=(), degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height =img.shape[0] + border * 2
    width = img.shape[1]
    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # apply angle-based reduction of bounding boxes
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        np.clip(xy, 0, height, out=xy)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets


def convert_images2bmp():
    # cv2.imread() jpg at 230 img/s, *.bmp at 400 img/s
    for path in ['../coco/images/val2014/', '../coco/images/train2014/']:
        folder = os.sep + Path(path).name
        output = path.replace(folder, folder + 'bmp')
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

        for f in tqdm(glob.glob('%s*.jpg' % path)):
            save_name = f.replace('.jpg', '.bmp').replace(folder, folder + 'bmp')
            cv2.imwrite(save_name, cv2.imread(f))

    for label_path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
        with open(label_path, 'r') as file:
            lines = file.read()
        lines = lines.replace('2014/', '2014bmp/').replace('.jpg', '.bmp').replace(
            '/Users/glennjocher/PycharmProjects/', '../')
        with open(label_path.replace('5k', '5k_bmp'), 'w') as file:
            file.write(lines)
