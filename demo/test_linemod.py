# @Time    : 02/04/2020 11:10
# @Author  : Wei Chen
# @Project : PyCharm


import argparse
from G2L_Net.yolov3.utils.utils import non_max_suppression
from G2L_Net.yolov3.models import Darknet, load_darknet_weights
from G2L_Net.utils.networks_usage import demo_linemod, load_models
from G2L_Net.utils.utils_funs import get_rotation, get_3D_corner, get_change_3D, depth_2_mesh_bbx, define_paras
import cv2

import numpy as np
import torch
from G2L_Net.utils import inout
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import yaml

def get4xy(cxr, cyr, wr, hr, w, h):
    # YOLO format: [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]

    x = cxr * w - (wr * w) / 2
    y = (cyr - hr / 2) * h

    return [x, y, x + wr * w, y, x + wr * w, y + hr * h, x, y + hr * h]


def getxywh(cxr, cyr, wr, hr, w, h, cen=1):
    if cen == 1:
        x = int(cxr * w - (wr * w) / 2)
        y = int((cyr - hr / 2) * h)
        W = int(wr * w)
        H = int(hr * h)
    else:
        x = int(cxr * w)
        y = int(cyr * h)
        W = int(wr * w)
        H = int(hr * h)

    return [x, y, W, H]


def getxyxy(cxr, cyr, wr, hr, w, h, cen=1):
    # YOLO format: [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]
    if cen == 1:
        x = int(cxr * w - (wr * w) / 2)
        y = int((cyr - hr / 2) * h)
        W = int(wr * w)
        H = int(hr * h)

        # print('tsts', x, y, W, H)

        x2 = x + W
        y2 = y + H
    else:
        x = int(cxr * w)
        y = int(cyr * h)
        W = int(wr * w)
        H = int(hr * h)

        # print('tsts', x, y, W, H)

        x2 = x + W
        y2 = y + H

    return [x, x2, y, y2]


def letterbox(img, height, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = max(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = max(0, (height[1] - new_shape[0]) / 2)  # width padding
    dh = max((height[0] - new_shape[1]) / 2, 0)  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, max(0, top), max(0, bottom), max(0, left), max(0, right), cv2.BORDER_CONSTANT,
                             value=color)  # padded square
    return img


def load_models_yolo(obj=15):
    cfg = '../yolov3/cfg/yolov3_test.cfg'

    model = Darknet(cfg)
    model.eval()
    model.cuda()
    weights = '../models/%d/best_%d.pt' % (obj, obj)  ## v2

    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    classifier, classifier_ce, classifier_box, classifier_box_gan, classifier_box_vec = load_models(obj, 199)

    model_path = '../models/%d/obj_%02d.ply' % (obj, obj)  ## m
    model2 = inout.load_ply(model_path)
    pc = model2['pts']

    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--data-cfg', type=str, default='cfg/obj.data', help='coco.data file path')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                        help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    opt = parser.parse_args()
    opt.cfg = cfg

    
    OR1, xr, yr, zr = get_3D_corner(pc)

    OR = get_change_3D(xr, yr, zr)
    OR_temp = OR

    OR[:, 0] = OR_temp[:, 0]
    OR[:, 1] = OR_temp[:, 1]
    OR[:, 2] = OR_temp[:, 2]

    base_path2 = '../models/'

    f = open(base_path2 + 'models_info.yml')
    temp = yaml.load(f.read(), Loader=yaml.FullLoader)
    f.close()
    return model, classifier, classifier_ce, classifier_box, classifier_box_gan, classifier_box_vec, pc, opt, OR, temp


def test(rgb, depth_, idx, model, classifier, classifier_ce, classifier_box, classifier_box_gan, classifier_box_vec, opt, pc, OR, Rt=0, Tt=0, step=1, imgid=0, temp =1):

    with torch.no_grad():

        img_size = opt.img_size

        conf_thres = opt.conf_thres
        nms_thres = opt.nms_thres

        CFG = define_paras()
        K = CFG['K']

        imgs = letterbox(rgb[0], [416, 416], color=(127.5, 127.5, 127.5))
        imgs = imgs[:, :, ::-1].transpose(2, 0, 1)
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)  # uint8 to float32
        imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        imgs = torch.from_numpy(imgs).unsqueeze(0).float()
        #

        output0, cors = model(imgs.cuda(), test=1)

        seen = 0

        output = non_max_suppression(output0, conf_thres=conf_thres, nms_thres=nms_thres)



        W = 640
        H = 480

        io = 0
        depth = depth_[io]
        pred = output[io]
        seen += 1
        if pred is None or np.any(depth) == None:

            print('no target object, please check')


        else:
            DC = int(W * cors[2][1][io] / 52)
            DR = int(H * cors[2][0][io] / 52)

            conf = pred[:, 4].cpu().numpy()
            xy42 = pred[np.argmax(conf)][0:4]

            xywh = getxyxy((xy42[0] + xy42[2]) / (img_size * 2), (xy42[1] + xy42[3]) / (img_size * 2),
                           (xy42[2] - xy42[0]) / img_size, (xy42[3] - xy42[1]) / img_size, W, H)


            rgb0 = rgb[io].copy()
            rgb01 = rgb[io]
            cv2.rectangle(rgb01, (xywh[0], xywh[2]), (xywh[1], xywh[3]), (255, 0, 0), 3)

            xywh2 = [xywh[2], xywh[3], xywh[0], xywh[1]]

            enl = 0
            xywh2[0] = max(xywh2[0] - enl, 0)
            xywh2[1] = min(xywh2[1] + enl, H)
            xywh2[2] = max(xywh2[2] - enl, 0)
            xywh2[3] = min(xywh2[3] + enl, W)

            cen_depth = np.zeros((1, 3))
            if depth[DR, DC] == 0:
                while depth[DR, DC] == 0:
                    DR = min(max(0, DR + np.random.randint(-10, 10)), 424)
                    DC = min(max(0, DC + np.random.randint(-10, 10)), 512)

            XC = [0, 0]
            XC[0] = np.float32(DC - K[0, 2]) * np.float32(depth[DR, DC] / K[0, 0])
            XC[1] = np.float32(DR - K[1, 2]) * np.float32(depth[DR, DC] / K[1, 1])

            cen_depth[0, 0:3] = [XC[0], XC[1], depth[DR, DC]]
            dep3d = depth_2_mesh_bbx(depth, xywh2, K)
            dep3d = dep3d[np.where(dep3d[:, 2] > 300.0)]

            def chooselimt_test(pts0, dia, cen):
                # cen = pts0.copy()
                pts = pts0.copy()
                pts = pts[np.where(pts[:, 2] > 20)[0], :]
                ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia)[0], :]
                if ptsn.shape[0] < 1000:
                    ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia * 2)[0], :]
                    if ptsn.shape[0] < 500:
                        ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia * 3)[0], :]
                return ptsn




            dep3d = chooselimt_test(dep3d, 102, cen_depth)  ##3 *N


            R = np.eye(3)
            T = 0
            if dep3d.shape[0] < 5:
                print('No enough valid depth points !!!')
            else:

                R, T =demo_linemod(dep3d, rgb0, rgb01, classifier, classifier_ce, classifier_box, classifier_box_gan, classifier_box_vec, pc, Rt, Tt, OR=OR, temp=temp)
            return R, T


if __name__ == '__main__':
    print('test')
    obj = 1
    model, classifier, classifier_ce, classifier_box, classifier_box_gan, classifier_box_vec, pc, opt, OR,temp = load_models_yolo(obj)
    val_list = '../models/%d/valseg.lst' % (obj)

    file_obj = open(val_list)
    all_lines = file_obj.readlines()
    file_obj.close()
    lists = []
    for line in all_lines:
        lists.append(line)


    base_path = '../test_sequence/01'


    Rts = np.loadtxt('../models/%d/R.txt' % (obj))
    Tts = np.loadtxt('../models/%d/T.txt' % (obj))
    step = 1
    C=0
    for il in range(0, len(lists)):
        print(il)
        idx = int(lists[il][-9:-1])

        Rt = Rts[idx * 3:(idx + 1) * 3, :]
        Tt = Tts[idx]

        rgbs = []
        deps = []

        idxx = idx


        rgbp = base_path + '/rgb/%04d.png' % (idxx)
        depthp = base_path + '/depth/%04d.png' % (idxx)

        rgb = cv2.imread(rgbp)
        rgbs.append(rgb)

        depth = cv2.imread(depthp, -1)
        deps.append(depth)

        R, T = test(rgbs, deps, idx, model, classifier, classifier_ce, classifier_box, classifier_box_gan, classifier_box_vec, opt, pc, OR, Rt, Tt, imgid=idxx, temp=temp)

