# @Time    : 06/03/2020 17:00
# @Author  : Wei Chen
# @Project : PyCharm
import numpy as np
import cv2

import ctypes as ct
import math
import os
import sys
import torch

def get_corners(obj, temp,num=8):
    obj=int(obj)
    minx = temp[obj]['min_x']
    miny = temp[obj]['min_y']
    minz = temp[obj]['min_z']
    sizex = temp[obj]['size_x']
    sizey = temp[obj]['size_y']
    sizez = temp[obj]['size_z']

    maxx=minx+sizex
    maxy=miny+sizey
    maxz=minz+sizez


    corners=np.zeros((8,3),dtype=np.float32)

    corners[3,:]=np.array([minx,miny,minz])
    corners[0, :] = np.array([minx, miny, maxz])
    corners[2, :] = np.array([minx, maxy, minz])
    corners[1, :] = np.array([minx, maxy, maxz])

    corners[4, :] = np.array([maxx, miny, minz])
    corners[5, :] = np.array([maxx, miny, maxz])
    corners[6, :] = np.array([maxx, maxy, minz])
    corners[7, :] = np.array([maxx, maxy, maxz])
    return corners[::8//num,:]


def define_paras():
    CFG={}



    K = np.array([[572.4114, 0, 325.2611],
                  [0, 573.57043,242.04899],
                  [0, 0, 1]])

    CFG['K']=K

    return CFG

def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(P.T, Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    U, S, V = np.linalg.svd(C)

    d = (np.linalg.det(V.T) * np.linalg.det(U.T)) <0.0



    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    E=np.diag(np.array([1,1,1]))

    # Create Rotation matrix U
    R = np.dot(V.T ,np.dot(E,U.T))

    return R

def gettrans(kps,h):

    hss=[]

    kps=kps.reshape(-1,3)
    for i in range(h.shape[1]):


        P = kps.T - kps.T.mean(1).reshape((3, 1))

        Q= h[:,i,:].T - h[:,i,:].T.mean(1).reshape((3,1))


        R=kabsch(P.T,Q.T)

        T=h[:,i,:]-np.dot(R,kps.T).T

        hh = np.zeros((3, 4), dtype=np.float32)
        hh[0:3,0:3]=R
        hh[0:3,3]=np.mean(T,0)

        hss.append(hh)

    return hss

def draw_cors_lite(img,K,R_,T_,color,OR, lindwidth=2):
    T_=T_.reshape((3,1))

    R=R_

    pcc=np.zeros((4,len(OR)),dtype='float32')
    pcc[0:3,:]=OR.T
    pcc[3,:]=1


    TT=np.zeros((3,4),dtype='float32')
    TT[:,0:3]=R
    TT[:,3]=T_[:,0]

    camMat=K

    pc_tt=np.dot(camMat,np.dot(TT,pcc))

    pc_t=np.transpose(pc_tt)
    x=pc_t[:,0]/pc_t[:,2]
    y=pc_t[:,1]/pc_t[:,2]





    cv2.line(img, (np.float32(x[0]),np.float32(y[0])), (np.float32(x[1]), np.float32(y[1])), color, lindwidth)
    cv2.line(img, (np.float32(x[1]),np.float32(y[1])), (np.float32(x[2]), np.float32(y[2])), color, lindwidth)
    cv2.line(img, (np.float32(x[2]),np.float32(y[2])), (np.float32(x[3]), np.float32(y[3])), color, lindwidth)
    cv2.line(img, (np.float32(x[3]),np.float32(y[3])), (np.float32(x[0]), np.float32(y[0])), color, lindwidth)

    cv2.line(img, (np.float32(x[0]),np.float32(y[0])), (np.float32(x[4]), np.float32(y[4])), color, lindwidth)
    cv2.line(img, (np.float32(x[1]),np.float32(y[1])), (np.float32(x[5]), np.float32(y[5])), color, lindwidth)
    cv2.line(img, (np.float32(x[2]),np.float32(y[2])), (np.float32(x[6]), np.float32(y[6])), color, lindwidth)
    cv2.line(img, (np.float32(x[3]),np.float32(y[3])), (np.float32(x[7]), np.float32(y[7])), color, lindwidth)

    cv2.line(img, (np.float32(x[4]),np.float32(y[4])), (np.float32(x[5]), np.float32(y[5])), color, lindwidth)
    cv2.line(img, (np.float32(x[5]),np.float32(y[5])), (np.float32(x[6]), np.float32(y[6])), color, lindwidth)
    cv2.line(img, (np.float32(x[6]),np.float32(y[6])), (np.float32(x[7]), np.float32(y[7])), color, lindwidth)
    cv2.line(img, (np.float32(x[7]),np.float32(y[7])), (np.float32(x[4]), np.float32(y[4])), color, lindwidth)

    return img

def get_rotation(x_,y_,z_):


    x=float(x_/180)*math.pi
    y=float(y_/180)*math.pi
    z=float(z_/180)*math.pi
    R_x=np.array([[1, 0, 0 ],
                 [0, math.cos(x), -math.sin(x)],
                 [0, math.sin(x), math.cos(x)]])

    R_y=np.array([[math.cos(y), 0, math.sin(y)],
                 [0, 1, 0],
                 [-math.sin(y), 0, math.cos(y)]])

    R_z=np.array([[math.cos(z), -math.sin(z), 0 ],
                 [math.sin(z), math.cos(z), 0],
                 [0, 0, 1]])
    return np.dot(R_z,np.dot(R_y,R_x))


showsz=500
mousex,mousey=0.5,0.5
zoom=1.0
changed=True
def onmouse(*args):
    global mousex,mousey,changed
    y=args[1]
    x=args[2]
    mousex=x/float(showsz)
    mousey=y/float(showsz)
    changed=True
def showpoints(xyz,c_gt=None, c_pred = None ,waittime=0,showrot=False,magnifyBlue=0,freezerot=False,background=(0,0,0),normalizecolor=True,ballradius=10):

    ## borrow from https://github.com/fxia22/pointnet.pytorch

    cv2.namedWindow('3D_seg')
    cv2.moveWindow('3D_seg', 1000, 500)
    cv2.setMouseCallback('3D_seg', onmouse)

    dll = np.ctypeslib.load_library('../utils/render_balls_so', '.')

    global showsz,mousex,mousey,zoom,changed

    xyz=xyz-xyz.mean(axis=0)

    xyz[:,2]=-xyz[:,2]
    xyz0=xyz.copy()
    xyz[:, 1] = xyz0[:, 0]
    xyz[:, 0] = xyz0[:, 1]

    radius=((xyz**2).sum(axis=-1)**0.5).max()

    xyz/=(radius*2.2)/showsz
    R = get_rotation(0,20,0)
    xyz = np.dot(R, xyz.T).T
    if c_gt is None:
        c0=np.zeros((len(xyz),),dtype='float32')+255
        c1=np.zeros((len(xyz),),dtype='float32')+255
        c2=np.zeros((len(xyz),),dtype='float32')+255
    else:
        c0=c_gt[:,0]
        c1=c_gt[:,1]
        c2=c_gt[:,2]


    if normalizecolor:
        c0/=(c0.max()+1e-14)/255.0
        c1/=(c1.max()+1e-14)/255.0
        c2/=(c2.max()+1e-14)/255.0


    c0=np.require(c0,'float32','C')
    c1=np.require(c1,'float32','C')
    c2=np.require(c2,'float32','C')

    show=np.zeros((showsz,showsz,3),dtype='uint8')
    def render():
        rotmat=np.eye(3)
        if not freezerot:
            xangle=(mousey-0.5)*np.pi*1.2
        else:
            xangle=0
        rotmat=rotmat.dot(np.array([
            [1.0,0.0,0.0],
            [0.0,np.cos(xangle),-np.sin(xangle)],
            [0.0,np.sin(xangle),np.cos(xangle)],
            ]))
        if not freezerot:
            yangle=(mousex-0.5)*np.pi*1.2
        else:
            yangle=0
        rotmat=rotmat.dot(np.array([
            [np.cos(yangle),0.0,-np.sin(yangle)],
            [0.0,1.0,0.0],
            [np.sin(yangle),0.0,np.cos(yangle)],
            ]))
        rotmat*=zoom
        nxyz=xyz.dot(rotmat)+[showsz/2,showsz/2,0]

        ixyz=nxyz.astype('int32')
        show[:]=background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue>0:
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
        if showrot:
            cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(30,showsz-30),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(30,showsz-50),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(30,showsz-70),0,0.5,cv2.cv.CV_RGB(255,0,0))
    changed=True
    while True:
        if changed:
            render()
            changed=False
        cv2.imshow('3D_seg',show)

        if waittime==0:
            cmd=cv2.waitKey(10)%256
        else:
            cmd=cv2.waitKey(waittime)%256


        if cmd==ord('q'):
            break
        elif cmd==ord('Q'):
            sys.exit(0)

        if cmd==ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0=np.zeros((len(xyz),),dtype='float32')+255
                    c1=np.zeros((len(xyz),),dtype='float32')+255
                    c2=np.zeros((len(xyz),),dtype='float32')+255
                else:
                    c0=c_gt[:,0]
                    c1=c_gt[:,1]
                    c2=c_gt[:,2]
            else:
                if c_pred is None:
                    c0=np.zeros((len(xyz),),dtype='float32')+255
                    c1=np.zeros((len(xyz),),dtype='float32')+255
                    c2=np.zeros((len(xyz),),dtype='float32')+255
                else:
                    c0=c_pred[:,0]
                    c1=c_pred[:,1]
                    c2=c_pred[:,2]
            if normalizecolor:
                c0/=(c0.max()+1e-14)/255.0
                c1/=(c1.max()+1e-14)/255.0
                c2/=(c2.max()+1e-14)/255.0
            c0=np.require(c0,'float32','C')
            c1=np.require(c1,'float32','C')
            c2=np.require(c2,'float32','C')
            changed = True



        if cmd==ord('n'):
            zoom*=1.1
            changed=True
        elif cmd==ord('m'):
            zoom/=1.1
            changed=True
        elif cmd==ord('r'):
            zoom=1.0
            changed=True
        elif cmd==ord('s'):
            cv2.imwrite('show3d.png',show)
        if waittime!=0:
            break
    return show

def get_3D_corner(pc):

    x_r=max(pc[:,0])-min(pc[:,0])
    y_r=max(pc[:,1])-min(pc[:,1])
    z_r=max(pc[:,2])-min(pc[:,2])


    ext1=np.array([0,x_r,y_r,z_r])
    or1=np.array([-ext1[1]/2,-ext1[2]/2,ext1[3]/2])
    or2=np.array([ext1[1]/2,-ext1[2]/2,ext1[3]/2])
    or3=np.array([ext1[1]/2,ext1[2]/2,ext1[3]/2])
    or4=np.array([-ext1[1]/2,ext1[2]/2,ext1[3]/2])

    or5=np.array([-ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or6=np.array([ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or7=np.array([ext1[1]/2,ext1[2]/2,-ext1[3]/2])
    or8=np.array([-ext1[1]/2,ext1[2]/2,-ext1[3]/2])

    OR=np.array([or1,or2,or3,or4,or5,or6,or7,or8])

    return OR, x_r,y_r,z_r

def get_change_3D(x_r,y_r,z_r):
    ext1=np.array([0,x_r,y_r,z_r])
    or1=np.array([-ext1[1]/2,-ext1[2]/2,ext1[3]/2])
    or2=np.array([ext1[1]/2,-ext1[2]/2,ext1[3]/2])
    or3=np.array([ext1[1]/2,ext1[2]/2,ext1[3]/2])
    or4=np.array([-ext1[1]/2,ext1[2]/2,ext1[3]/2])

    or5=np.array([-ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or6=np.array([ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or7=np.array([ext1[1]/2,ext1[2]/2,-ext1[3]/2])
    or8=np.array([-ext1[1]/2,ext1[2]/2,-ext1[3]/2])

    OR=np.array([or1,or2,or3,or4,or5,or6,or7,or8])
    return OR


def depth_2_mesh_bbx(depth,bbx,K, step=1, dr=0,dc=0):


    x1 = int(max(bbx[0],0))
    x2 = int(min(bbx[1],depth.shape[0]))
    y1 = int(max(bbx[2],0))
    y2 = int(min(bbx[3],depth.shape[1]))


    mesh = depth_2_pc(depth, K, bbx = [x1,x2,y1,y2], step=step, dr=dr,dc=dc)


    return mesh

def depth_2_pc(depth, K, bbx=[1, 2, 3, 4], step=1, dr=0, dc=0):
    x1 = bbx[0]
    x2 = bbx[1]
    y1 = bbx[2]
    y2 = bbx[3]

    fx = K[0, 0]
    ux = K[0, 2]
    fy = K[1, 1]
    uy = K[1, 2]
    W = y2 - y1 + 1
    H = x2 - x1 + 1

    xw0 = np.arange(y1, y2, step)
    xw0 = np.expand_dims(xw0, axis=0)
    xw0 = np.tile(xw0.T, 2)
    uu0 = np.zeros_like(xw0, dtype=np.float32)
    uu0[:, 0] = ux
    uu0[:, 1] = uy

    mesh = np.zeros((len(range(0, H, step)) * xw0.shape[0], 3))
    c = 0
    for i in range(x1, x2, step):
        xw = xw0.copy()
        uu = uu0.copy()
        xw[:, 0] = i

        z = depth[xw[:, 0], xw[:, 1]]

        xw[:, 0] = xw[:, 0] * z
        xw[:, 1] = xw[:, 1] * z

        uu[:, 0] = uu[:, 0] * z
        uu[:, 1] = uu[:, 1] * z

        X = (xw[:, 1] - uu[:, 0]) / fx
        Y = (xw[:, 0] - uu[:, 1]) / fy
        mesh[xw.shape[0] * c:xw.shape[0] * (c + 1), 0] = X
        mesh[xw.shape[0] * c:xw.shape[0] * (c + 1), 1] = Y
        mesh[xw.shape[0] * c:xw.shape[0] * (c + 1), 2] = z
        c += 1

    return mesh


def getFiles_ab(file_dir,suf,a,b):
    L=[]
    for root, dirs, files in os.walk(file_dir):

        for file in files:
            if os.path.splitext(file)[1] == suf:

                L.append(os.path.join(root, file))

        L.sort(key=lambda x: int(x[a:b]))
    return L


def getFiles_any(file_dir,suf):
    L=[]
    for  file in os.listdir(file_dir):

        if os.path.splitext(file)[1] == suf:


            L.append(os.path.join(file))


    return L


def draw_mesh_3D(ax,pc, R_,T_,fl,color,s2=10,alpha=1):


    '''

    :param ax:
    :param pc: N*3
    :param R_:
    :param T_:
    :param fl:
    :param color:
    :return:
    '''
    pc=pc.reshape((-1,3))
    if fl==1:
        RR=(cv2.Rodrigues(R_))
        R=RR[0]
    else:
        R=R_


    R_m=get_rotation(0,0,0)

    R=np.dot(R,R_m)
    pc_temp=pc

    pc[:,0]=pc_temp[:,0]
    pc[:,1]=pc_temp[:,1]
    pc[:,2]=pc_temp[:,2]

    pcc=np.zeros((4,len(pc)),dtype='float32')
    pcc[0:3,:]=pc.T
    pcc[3,:]=1


    TT=np.zeros((3,4),dtype='float32')
    TT[:,0:3]=R
    TT[:,3]=T_


    pc_tt=np.dot(TT,pcc)


    ax.scatter(pc_tt[0,:],pc_tt[1,:],pc_tt[2,:],c=color,s=s2,alpha=alpha)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def seg2txt(seg,idx,rgb_path):
    H = seg.shape[0]
    W = seg.shape[1]
    r, c = np.where(seg > 0)
    enl = 0
    xmin = max(c.min() - enl, 0)
    ymin = max(r.min() - enl, 0)
    wid = min(c.max() + enl, W) - c.min()
    hig = min(r.max() + enl, H) - r.min()

    b1 = (xmin + wid / 2) / W
    b2 = (ymin + hig / 2) / H
    b3 = wid / W
    b4 = hig / H
    txt_name = '%04d.txt'%(idx)
    txtnames = rgb_path + txt_name

    fid = open(txtnames, 'w+')
    fid.write('%d ' % (0))
    fid.write('%f ' % (b1))
    fid.write('%f ' % (b2))
    fid.write('%f ' % (b3))
    fid.write('%f ' % (b4))
    fid.write('\n')

    fid.close()

def trans_3d(pc,Rt,Tt):

    '''

    :param pc: should be n*3
    :param Rt: 3*3
    :param Tt: 3*1
    :return:
    '''

    Tt=np.reshape(Tt,(3,1))
    pcc=np.zeros((4,pc.shape[0]),dtype=np.float32)
    pcc[0:3,:]=pc.T
    pcc[3,:]=1


    TT=np.zeros((3,4),dtype=np.float32)
    TT[:,0:3]=Rt
    TT[:,3]=Tt[:,0]

    trans=np.dot(TT,pcc)


    return trans  ## 3 N
def get6dpose2_f(pcc,Rt,Tt,R,T):
    estPt = trans_3d(pcc, R, T)

    gtPt0 = trans_3d(pcc, Rt, Tt)
    delta = estPt - gtPt0

    ss=np.linalg.norm(delta,2, 0)
    dis=sum(ss.flatten())/pcc.shape[0]

    return dis

def read_RT(obj):
    base_path = '../models/%d/' % (obj)

    R=np.loadtxt(base_path+'R.txt')
    T=np.loadtxt(base_path+'T.txt')
    return R, T

def get_vectors(pts, kps, vn=1):
    '''

    :param pts: N*3
    :param kps: M*3
    :return:
    '''

    ## N M 3
    pts.view(-1,3)


    kps=kps.view(-1,3)
    N=pts.shape[0]


    M=kps.shape[0]


    pts=pts.unsqueeze(1).repeat(1,M,1)## N M 3
    kps=kps.unsqueeze(0).repeat(N,1,1)## N M 3
    vecs=kps-pts.float() ## N M 3

    if vn==1:
        vn=vecs.norm(dim=2, keepdim=True) ## N M 1
        vn=vn.repeat(1,1,3) ## N M 3
        vecs=vecs/vn ## N M 3
        return vecs.view(pts.shape[0], -1)  ## N M*3
    else:
        return vecs.view(pts.shape[0], -1)  ## N M*3
def data_augment(points, Rs, Ts, obj_id, temp, num_c, target_seg, idxs,ax=5, ay=5, az=25, a=10):

    centers = np.zeros((points.shape[0], 3))
    corners = np.zeros((points.shape[0], 3 * num_c))
    vecs = torch.zeros(points.shape[0], points.shape[1], 3*num_c)
    pts0 = points.copy()
    for ii in range(points.shape[0]):

        idx = idxs[ii].item()

        Rt = Rs[idx * 3:(idx + 1) * 3, 0:3]
        Tt = Ts[idx]

        res = np.mean(points[ii], 0)
        points[ii, :, 0:3] = points[ii, :, 0:3] - np.array([res[0], res[1], res[2]])



        dx = np.random.randint(-ax, ax)
        dy = np.random.randint(-ay, ay)
        dz = np.random.randint(-az, az)

        points[ii, :, 0] = points[ii, :, 0] + dx
        points[ii, :, 1] = points[ii, :, 1] + dy
        points[ii, :, 2] = points[ii, :, 2] + dz




        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))

        points[ii, :, 0:3] = np.dot(Rm, points[ii, :, 0:3].T).T

        pts_seg = pts0[ii, np.where(target_seg.numpy()[ii, :] == 1)[0], 0:3]
        centers[ii,:]=Tt.T-np.mean(pts_seg,0)


        Tt_c = np.array([0, 0, 0]).T
        corners_ = get_corners(obj_id[ii].numpy(), temp, num=num_c)

        pts_noT = pts_seg - Tt.T


        pts_noT =np.dot(Rm, pts_noT.T).T


        corners[ii, :] = (trans_3d(corners_ , np.dot(Rm, Rt), Tt_c).T).flatten()
        vecs[ii, np.where(target_seg[ii, :] == 1)[0]] = get_vectors(torch.from_numpy(pts_noT), torch.from_numpy(corners[ii, :]).float())


    return points, corners, centers, vecs
