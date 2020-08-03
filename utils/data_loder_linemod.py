# @Time    : 15/04/2020 12:06
# @Author  : Wei Chen
# @Project : PyCharm
from torch.utils.data import Dataset, DataLoader
import numpy as np



def chooselimt(pts0, dia, lab):
    a = pts0[lab[:,0] == 1, :]
    pts = pts0.copy()
    ptss=pts[lab[:,0]==1,:]
    idx = np.random.randint(0,a.shape[0])
    ptsn=pts[np.where(np.abs(pts[:,2]-ptss[idx,2].max())<dia)[0],:]
    labs = lab[np.where(np.abs(pts[:, 2] - ptss[idx, 2].max()) < dia)[0],:]

    return ptsn,labs

class ObjDataset_all(Dataset):
    def __init__(self, labels, root_dir, temp,obj):

        self.root_dirlab = labels

        self.root_dir=root_dir

        self.rad=temp[obj]['diameter']
        self.obj= obj
        train_list = '../models/%d/train.lst' % (obj)

        file_obj = open(train_list)
        all_lines = file_obj.readlines()
        file_obj.close()
        lists = []
        for line in all_lines:
            lists.append(line)

        self.indexs = lists

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx):


        ref_id = int(self.indexs[idx])


        lab_name = self.root_dirlab + 'lab%08d.txt' % (ref_id)
        pts_name = self.root_dir + 'pose%08d.txt' % (ref_id)

        label = np.loadtxt(lab_name)

        label = label[:,0].reshape((-1, 1))



        points = np.loadtxt(pts_name)


        assert points.shape[0]==label.shape[0]


        points,label=chooselimt(points,self.rad,label)


        choice = np.random.choice(len(points), 1500, replace=True)
        points = points[choice, :]
        label = label[choice, :]

        sample = {'points': points, 'label': label, 'obj': self.obj,'idx': ref_id}

        return sample

def load_pts_train(datas_list, labs_p ,bat, temp,obj,shuf=True,drop=False):

    data=ObjDataset_all(labs_p,datas_list,temp,obj)


    dataloader = DataLoader(data, batch_size=bat, shuffle=shuf, drop_last=drop)


    return dataloader
