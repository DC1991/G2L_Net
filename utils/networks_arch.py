# @Time    : 06/03/2020 17:37
# @Author  : Wei Chen
# @Project : PyCharm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class Point_seg(nn.Module):
    def __init__(self, ic = 6):
        super(Point_seg, self).__init__()
        self.conv1 = torch.nn.Conv2d(ic, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 64, 1)

        self.conv3 = torch.nn.Conv2d(64, 64, 1)
        self.conv4 = torch.nn.Conv2d(64, 128, 1)

        self.conv5 = torch.nn.Conv2d(128, 1024, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)


    def forward(self, x):
        n_pts = x.size()[3]

        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.max(x, -2, keepdim=True)[0]
        x1 = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x1)))
        x = F.relu(self.bn4(self.conv4(x)))
        x2 = torch.max(x, -2, keepdim=True)[0]
        x = self.bn5(self.conv5(x2))
        x = torch.max(x, -2, keepdim=True)[0]
        pointfeat2 = x
        x = torch.max(x, -1, keepdim=True)[0]

        x = x.view(-1, 1024, 1,1 ).repeat(1, 1, 1, n_pts)
        return torch.cat([x1,x2 ,x, pointfeat2], 1)

class Seg_3D(nn.Module):
    def __init__(self,kc=2,ic=6):
        super(Seg_3D, self).__init__()

        self.kc = kc
        self.feat = Point_seg(ic=ic)
        self.conv1 = torch.nn.Conv2d(1024+1024+128+64+16, 512, 1)
        self.conv2 = torch.nn.Conv2d(512, 256, 1)
        self.conv3 = torch.nn.Conv2d(256, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 128, 1)
        self.conv5 = torch.nn.Conv2d(128, self.kc, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, x, obj):
        batchsize = x.size()[0]
        n_pts = x.size()[3]
        x = self.feat(x)

        if obj.shape[0] == 1:
            obj = obj.view(-1, 1).repeat(batchsize, 1)
        else:
            obj = obj.view(-1, 1)

        one_hot = torch.zeros(batchsize, 16).scatter_(1, obj.cpu().long(), 1)
        one_hot=one_hot.cuda()
        one_hot2=one_hot.unsqueeze(2).repeat(1,1,n_pts)
        one_hot3 = one_hot.unsqueeze(2).unsqueeze(3).repeat(1, 1,1, n_pts)
        feas = x.squeeze(2)

        feas=torch.cat([feas,one_hot2],1)
        x = torch.cat([x, one_hot3], 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x=self.drop1(x)
        x = self.conv5(x)

        x=x.squeeze(2)

        x = x.transpose(2,1).contiguous()

        x_seg=x
        x_seg = F.log_softmax(x_seg.view(-1, self.kc), dim=-1)


        x_seg = x_seg.view(batchsize, n_pts, self.kc)


        return x_seg



class Point_center(nn.Module):
    def __init__(self):
        super(Point_center, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)


        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)


        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)


    def forward(self, x,obj):## 5 6 30 1000
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = (self.bn3(self.conv3(x)))

        x2 = torch.max(x, -1, keepdim=True)[0]

        obj = obj.view(-1, 1)
        one_hot = torch.zeros(batchsize, 16).scatter_(1, obj.cpu().long(), 1)

        one_hot = one_hot.cuda()
        one_hot2 = one_hot.unsqueeze(2)
        return torch.cat([x2, one_hot2],1)


class Point_center_res(nn.Module):
    def __init__(self):
        super(Point_center_res, self).__init__()

        self.feat = Point_center()
        self.conv1 = torch.nn.Conv1d(512+16, 256,1)
        self.conv2 = torch.nn.Conv1d(256, 128,1)

        self.conv3 = torch.nn.Conv1d(128, 3,1 )


        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.2)

    def forward(self, x, obj):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        x = self.feat(x, obj)



        x = F.relu(self.bn1(self.conv1(x)))
        x = (self.bn2(self.conv2(x)))

        x=self.drop1(x)
        x = self.conv3(x)



        x = x.squeeze(2)
        x=x.contiguous()

        return x



class Point_box_v(nn.Module):
    def __init__(self,inputchannel=6):
        super(Point_box_v, self).__init__()

        self.conv1 = torch.nn.Conv2d(inputchannel, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 64, 1)

        self.conv3 = torch.nn.Conv2d(64, 64, 1)
        self.conv4 = torch.nn.Conv2d(64, 128, 1)

        self.conv5 = torch.nn.Conv2d(128, 1024, 1)


        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)


    def forward(self, x, obj):

        n_pts = x.size()[3]

        x = F.relu(self.bn1(self.conv1(x)))

        x = torch.max(x, -2, keepdim=True)[0]

        x1 = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x1)))
        x = F.relu(self.bn4(self.conv4(x)))
        x2 = torch.max(x, -2, keepdim=True)[0]

        x = self.bn5(self.conv5(x2))
        x = torch.max(x, -2, keepdim=True)[0]
        pointfeat2 = x

        x = torch.max(x, -1, keepdim=True)[0]



        x = x.view(-1, 1024, 1, 1).repeat(1, 1, 1, n_pts)
        return torch.cat([x1, x2, x, pointfeat2], 1)



class Point_box_v_es(nn.Module):
    def __init__(self, num=8, inputchannel=6):
        super(Point_box_v_es, self).__init__()


        self.feat = Point_box_v(inputchannel)


        self.conv1 = torch.nn.Conv2d(1024 + 1024 + 128 + 64 + 16, 512, 1)

        self.conv2 = torch.nn.Conv2d(512, 256, 1)
        self.conv3 = torch.nn.Conv2d(256, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 128, 1)
        self.conv5 = torch.nn.Conv2d(128, 3*num, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.drop1 = nn.Dropout(0.2)
    def forward(self, x,obj, gan=0):
        batchsize = x.size()[0]
        n_pts = x.size()[3]


        if obj.shape[0]==1:
            obj = obj.view(-1, 1).repeat(batchsize,1)
        else:
            obj = obj.view(-1, 1)

        x = self.feat(x, obj)
        feat = x.detach().squeeze(-1)

        one_hot = torch.zeros(batchsize, 16).scatter_(1, obj.cpu().long(), 1)
        one_hot = one_hot.cuda()
        one_hot2 = one_hot.unsqueeze(2).repeat(1, 1, n_pts)
        one_hot3 = one_hot.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, n_pts)
        feas = feat.squeeze(2)

        feas = torch.cat([feas, one_hot2], 1)
        x = torch.cat([x, one_hot3], 1)
        x = F.relu(self.bn1(self.conv1(x)))



        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop1(x)
        x = self.conv5(x)

        x = x.squeeze(2)

        x = x.transpose(2, 1).contiguous()



        if gan==1:
            return x, feas
        else:
            return x


class Point_box_R_es(nn.Module):
    def __init__(self, num=8):
        super(Point_box_R_es, self).__init__()


        self.drop0 = nn.Dropout(0.3)
        self.conv1 = torch.nn.Conv1d(2256, 256,1)
        self.conv2 = torch.nn.Conv1d(256, 128,1)
        self.conv3 = torch.nn.Conv1d(128, 3*num,1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)


    def forward(self, x):

        x = self.drop0(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = (self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = ((self.conv3(x)))
        x = x.contiguous()
        return x


class Rotation_pre(nn.Module):
    def __init__(self, k=24,F=2256):
        super(Rotation_pre, self).__init__()
        self.f=F
        self.k = k

        self.conv1 = torch.nn.Conv1d(self.f , 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 256,1)
        self.conv5 = torch.nn.Conv1d(256, 128,1)
        self.conv6 = torch.nn.Conv1d(128, 128,1)
        self.conv7 = torch.nn.Conv1d(128,self.k,1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x=self.drop1(x)
        x = self.conv7(x)

        x=x.squeeze(2)
        x = x.contiguous()


        return x