# @Time    : 03/08/2020 18:18
# @Author  : Wei Chen
# @Project : Pycharm

from __future__ import print_function

import os
import argparse
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from G2L_Net.utils.networks_arch import *
from G2L_Net.utils.utils_funs import read_RT, data_augment, get_corners, get_RT_bat
import torch.nn.functional as F
from G2L_Net.utils.data_loder_linemod import load_pts_train
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--outk', type=int, default=14, help='vector and class')
parser.add_argument('--outclass', type=int, default=2, help='point class')


opt = parser.parse_args()
k = opt.outk
kc = opt.outclass
num_c = 8
batch_size = opt.batchsize

for obj in [6]:

    classifier = Seg_3D(kc=kc, ic=3)  ##3d
    classifier_ce = Point_center_res()

    classifier_box = Point_box_v_es(num_c, inputchannel=3)  ## evf
    classifier_box_gan = Point_box_R_es(num_c)  ## rotation residual
    classifier_box_vec = Rotation_pre(F=2256 + 3 * num_c, k=3 * num_c)  ### rotation prediction with vectors & EVF

    num_classes = opt.outclass

    Loss_func_ce = nn.MSELoss()
    Loss_func_box = nn.MSELoss()
    Loss_func_box_gan = nn.MSELoss()
    Loss_func_box_vec = nn.MSELoss()
    Loss_func_ce.cuda()
    Loss_func_box.cuda()
    Loss_func_box_gan.cuda()
    Loss_func_box_vec.cuda()

    classifier = nn.DataParallel(classifier)
    classifier_ce = nn.DataParallel(classifier_ce)
    classifier_box = nn.DataParallel(classifier_box)
    classifier_box_gan = nn.DataParallel(classifier_box_gan)
    classifier_box_vec = nn.DataParallel(classifier_box_vec)

    classifier.cuda()
    classifier = classifier.train()

    classifier_ce.cuda()
    classifier_ce = classifier_ce.train()

    classifier_box.cuda()
    classifier_box = classifier_box.train()

    classifier_box_gan.cuda()
    classifier_box_gan = classifier_box_gan.train()

    classifier_box_vec.cuda()
    classifier_box_vec = classifier_box_vec.train()

    opt.outf = '../models/%d/trained/'%(obj)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    data_path = '../train_data/%d/'%(obj)

    train_pts = data_path + 'pts/'
    train_lab = data_path + 'pts_lab/'

    lr = 0.001

    epochs = opt.nepoch
    base_path2 = '../models/'

    f = open(base_path2 + 'models_info.yml')
    Dia = yaml.load(f.read(), Loader=yaml.FullLoader)
    f.close()

    Rs, Ts = read_RT(obj)
    optimizer = optim.Adam([{'params': classifier.parameters()},{'params': classifier_ce.parameters(), 'lr': 1e-3*2},{'params': classifier_box.parameters(),'lr': 1e-3*2},{'params': classifier_box_gan.parameters(),'lr': 1e-3*2},{'params': classifier_box_vec.parameters(),'lr': 1e-3*2}], lr=lr, betas=(0.9, 0.99))

    dataloader = load_pts_train(train_pts, train_lab, batch_size, Dia, obj)

    for epoch in range(epochs):

        if epoch > 0 and epoch % (epochs // 4) == 0:
            lr = lr / 4

        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * 10
        optimizer.param_groups[2]['lr'] = lr
        optimizer.param_groups[3]['lr'] = lr * 20
        optimizer.param_groups[4]['lr'] = lr * 30



        for i, data in enumerate(dataloader):

            points, target_, obj_id, idxs = data['points'], data['label'], data['obj'], data['idx']
            ptsori = points.clone()

            target_seg = target_[:, :, 0]

            points_ = ptsori.numpy().copy()


            points_a, corners, centers, vecs = data_augment(points_, Rs, Ts, obj_id, Dia, num_c, target_seg, idxs,ax=5, ay=5,az=25, a=10.0)


            points, target_seg, vecs = torch.from_numpy(points_a).cuda(), target_seg.cuda(), vecs.cuda()


            pointsf = points.unsqueeze(2)

            optimizer.zero_grad()

            points = pointsf.transpose(3, 1)

            pred_seg = classifier(points.float(), obj_id)

            pred_choice = pred_seg.data.max(2)[1]

            p = pred_choice
            N_seg = 1000
            pts_s = torch.zeros(points.shape[0], N_seg, 3)

            vecsseg = torch.zeros(points.shape[0], N_seg, 3 * num_c).cuda()
            corners0 = np.zeros((points.shape[0], num_c, 3))
            ptsori = ptsori.cuda()
            Tt = np.zeros((points.shape[0], 3))
            for ib in range(points.shape[0]):
                if len(p[ib, :].nonzero()[:, 0]) < 10:
                    continue

                pts_ = torch.index_select(ptsori, 0, p[ib, :].nonzero()[:, 0])  ##Nx3
                vecs_ = torch.index_select(vecs[ib, :, :], 0, p[ib, :].nonzero()[:, 0])
                choice = np.random.choice(len(pts_), N_seg, replace=True)
                pts_s[ib, :, :] = pts_[choice, :]
                vecsseg[ib, :, :] = vecs_[choice, :]
                corners0[ib] = get_corners(obj_id[ib].numpy(), Dia, num_c)


            corners0 = torch.Tensor(corners0).cuda()


            pts_s = pts_s.cuda()

            pts_s = pts_s.transpose(2, 1)
            cen_pred = classifier_ce((pts_s - pts_s.mean(dim=2, keepdim=True)), obj_id)


            B = points.shape[0]

            box_pt0 = (pts_s - pts_s.mean(dim=2, keepdim=True) - cen_pred.unsqueeze(2)).detach()

            box_pt = box_pt0.transpose(2, 1).unsqueeze(2)
            box_pt = box_pt.transpose(3, 1)

            box_pred, feat = classifier_box(box_pt, obj_id, gan=1)

            feavec = torch.cat([box_pred, feat.transpose(1, 2)], 2)
            feavec = feavec.transpose(1, 2)

            kp_m = classifier_box_vec(feavec)



            centers = Variable(torch.Tensor((centers)))
            centers = centers.cuda()

            corners = Variable(torch.Tensor((corners)))
            corners = corners.cuda()

            vecsseg = Variable(vecsseg)
            cors_box = get_RT_bat(corners0, kp_m.detach().view(box_pred.shape[0], -1, 3), a=0)

            cors_box = cors_box.contiguous().view(cors_box.shape[0], -1, 1)


            pred_seg = pred_seg.view(-1, num_classes)
            target_seg = target_seg.view(-1, 1)[:, 0]

            loss_seg = F.nll_loss(pred_seg, target_seg.long())
            loss_res = Loss_func_ce(cen_pred, centers.float())
            loss_box = Loss_func_box(box_pred.view(vecsseg.shape[0], -1, 3 * num_c), vecsseg.float())
            loss_box_vec = Loss_func_box_vec(kp_m, corners.float())





            box_pred_gan = classifier_box_gan(feat.squeeze(-2))
            box_pred_gan = box_pred_gan + cors_box.detach()
            loss_box_gan = Loss_func_box_gan(box_pred_gan.squeeze(2), corners.float())


            Loss = loss_seg*10.0+loss_res/10.0+loss_box*10.0+loss_box_gan/100.0+loss_box_vec/100.0
            Loss.backward()
            optimizer.step()

            print('[%d: %d] train loss_seg: %f, loss_res: %f, loss_box: %f,loss_box_vec: %f, loss_box_gan: %f' % (
            epoch, i, loss_seg.item(), loss_res.item(), loss_box.item(), loss_box_vec.item(), loss_box_gan.item()))


            print()
        if epoch % epochs == epochs - 1 or (epoch - 10) % 20 == 0:
            torch.save(classifier.state_dict(), '%s/Seg3D_epoch%d_obj%d.pth' % (opt.outf, epoch, obj))
            torch.save(classifier_ce.state_dict(), '%s/Tres_epoch%d_obj%d.pth' % (opt.outf, epoch, obj))
            torch.save(classifier_box.state_dict(), '%s/EVF_epoch%d_obj%d.pth' % (opt.outf, epoch, obj))
            torch.save(classifier_box_vec.state_dict(), '%s/RE_epoch%d_obj%d.pth' % (opt.outf, epoch, obj))
            torch.save(classifier_box_gan.state_dict(), '%s/R_res_E_epoch%d_obj%d.pth' % (opt.outf, epoch, obj))




