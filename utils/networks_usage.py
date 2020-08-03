# @Time    : 06/03/2020 17:02
# @Author  : Wei Chen
# @Project : PyCharm
import numpy as np
import torch
from torch.autograd import Variable
from G2L_Net.utils.utils_funs import get_corners
import cv2
from G2L_Net.utils.utils_funs import define_paras,gettrans,draw_cors_lite,showpoints,get6dpose2_f
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import argparse
from G2L_Net.utils.networks_arch import Seg_3D, Point_center_res,Point_box_v_es,Point_box_R_es,Rotation_pre

def load_models(obj, epoch=199):

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=epoch, help='number of epochs to train for')
    parser.add_argument('--outf', type=str,
                        default='../models/%d'%(obj),
                        help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    opt = parser.parse_args()

    opt.outclass = 2
    kc = 2
    classifier_3D = Seg_3D(kc, ic=3)
    classifier_ce = Point_center_res()


    num_c = 8
    classifier_box = Point_box_v_es(num_c, inputchannel=3)
    classifier_box_gan = Point_box_R_es(num_c)

    classifier_box_vec = Rotation_pre(F=2280)


    Loss_func_ce = nn.MSELoss()
    Loss_func_box = nn.MSELoss()

    Loss_func_ce.cuda()
    Loss_func_box.cuda()


    classifier = nn.DataParallel(classifier_3D)
    classifier.cuda()
    classifier = classifier.eval()

    classifier_ce = nn.DataParallel(classifier_ce)
    classifier_ce.cuda()
    classifier_ce = classifier_ce.eval()

    classifier_box = nn.DataParallel(classifier_box)
    classifier_box.cuda()
    classifier_box = classifier_box.eval()

    classifier_box_gan = nn.DataParallel(classifier_box_gan)
    classifier_box_gan.cuda()
    classifier_box_gan = classifier_box_gan.eval()

    classifier_box_vec = nn.DataParallel(classifier_box_vec)
    classifier_box_vec.cuda()
    classifier_box_vec = classifier_box_vec.eval()

    objm = obj

    model_class = '%s/Seg3D_epoch%d_obj%d.pth' % (opt.outf, epoch, objm)
    classifier.load_state_dict(torch.load(model_class))


    model_res = '%s/Tres_epoch%d_obj%d.pth' % (
        opt.outf,  epoch, objm)
    classifier_ce.load_state_dict(torch.load(model_res))

    model_box = '%s/EVF_epoch%d_obj%d.pth' % (opt.outf, epoch, objm)
    classifier_box.load_state_dict(torch.load(model_box))

    model_box_RRes = '%s/R_res_E_epoch%d_obj%d.pth' % (
        opt.outf, epoch, objm)
    classifier_box_gan.load_state_dict(torch.load(model_box_RRes))

    model_box_R = '%s/RE_epoch%d_obj%d.pth' % (
        opt.outf, epoch, objm)
    classifier_box_vec.load_state_dict(torch.load(model_box_R))

    return classifier, classifier_ce, classifier_box, classifier_box_gan, classifier_box_vec

def demo_linemod(pts, rgb, rgb2,classifier,classifier_ce,classifier_box,classifier_box_gan,classifier_box_vec, pc, Rt=0, Tt=0, OR = 0, temp=0):
    numc = 8
    num_c = 8


    obj = 1





    points = torch.Tensor(pts).unsqueeze(0)
    ptsori = points.clone()



    obj_id = torch.Tensor([obj])

    points = points.numpy().copy()
    res = np.mean(points[0], 0)
    points[0, :, 0:3] = points[0, :, 0:3] - np.array([res[0], res[1], res[2]])

    corners_ = get_corners(obj, temp,num=num_c)

    points = Variable(torch.Tensor(points))

    points = points.cuda()

    pointsf = points[:, :, 0:3].unsqueeze(2)

    points = pointsf.transpose(3, 1)

    pred_seg = classifier(points, obj_id)

    pred_choice = pred_seg.data.max(2)[1]

    p = pred_choice
    cmap = plt.cm.get_cmap("hsv", 20)
    cmap = np.array([cmap(i) for i in range(20)])[:, :3]

    pred_choice1 = pred_choice.cpu().numpy()[0]
    pred_color0 = cmap[pred_choice1, :]

    ptsori = ptsori.cuda()

    if len(p[0, :].nonzero()[:, 0])<10:
        print('No object pts')
    else:
        pts_ = torch.index_select(ptsori[0, :, 0:3], 0, p[0, :].nonzero()[:, 0].cuda())  ##Nx3

        pts_s = pts_[:, :].unsqueeze(0).float()


        pts_s = pts_s.cuda()

        pts_s = pts_s.transpose(2, 1)
        cen_pred = classifier_ce((pts_s - pts_s.mean(dim=2, keepdim=True)), obj_id)
        box_pt0 = (pts_s - pts_s.mean(dim=2, keepdim=True) - cen_pred.unsqueeze(2)).detach()
        box_pt=box_pt0.transpose(2, 1).unsqueeze(2)
        box_pt = box_pt.transpose(3, 1)
        box_pred, feat = classifier_box(box_pt, obj_id, 1)


        corners_ = corners_.reshape((numc, 1, 3))


        feavec = torch.cat([box_pred, feat.transpose(1, 2)], 2)  ##
        feavec = feavec.transpose(1, 2)
        kp_m = classifier_box_vec(feavec.detach())



        pose = gettrans(corners_.reshape((numc, 3)), kp_m.view((numc, 1, 3)).detach().cpu().numpy())


        R = pose[0][0:3, 0:3]

        corners_ = corners_.reshape((numc, 3))
        cors_box = np.dot(R, corners_.T).T
        cors_box = torch.from_numpy(cors_box).cuda()
        #
        cors_box = cors_box.contiguous().view(1, -1, 1)

        box_pred_gan = classifier_box_gan(feat)
        box_pred_gan = box_pred_gan + cors_box

        pose_gan = gettrans(corners_.reshape((numc, 3)), box_pred_gan.view((numc, 1, 3)).detach().cpu().numpy())

        T = (pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2)).view(1, 3).detach().cpu().numpy()


        Rg = pose_gan[0][0:3, 0:3]



        return  Rg, T