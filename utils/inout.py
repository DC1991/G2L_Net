# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np
import struct
import cv2

def load_poses(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        poses = []
        for line in lines:
            if not line.isspace():
                elems = line.split(' ')
                im_id = int(elems[0])
                obj_id = int(elems[1])
                R = np.array(map(float, elems[2:11])).reshape((3, 3))
                t = np.array(map(float, elems[11:14])).reshape((3, 1))
                pose = {'im_id': im_id, 'obj_id': obj_id, 'R': R, 't': t}
                if len(elems) > 14:
                    pose['score'] = float(elems[14])
                poses.append(pose)
    return poses

def load_gt_pose_dresden(path):
    R = []
    t = []
    rotation_sec = False
    center_sec = False
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            if 'rotation:' in line:
                rotation_sec = True
            elif rotation_sec:
                R += line.split(' ')
                if len(R) == 9:
                    rotation_sec = False
            elif 'center:' in line:
                center_sec = True
            elif center_sec:
                t = line.split(' ')
                center_sec = False

    assert((len(R) == 0 and len(t) == 0) or
           (len(R) == 9 and len(t) == 3))

    if len(R) == 0:
        pose = {'R': np.array([]), 't': np.array([])}
    else:
        pose = {'R': np.array(list(map(float, R))).reshape((3, 3)),
                't': np.array(list(map(float, t))).reshape((3, 1))}

        # Flip Y and Z axis (OpenGL -> OpenCV coordinate system)
        yz_flip = np.eye(3, dtype=np.float32)
        yz_flip[0, 0], yz_flip[1, 1], yz_flip[2, 2] = 1, -1, -1
        pose['R'] = yz_flip.dot(pose['R'])
        pose['t'] = yz_flip.dot(pose['t'])
    return pose

def load_ply(path):
    """
    Loads a 3D mesh model from a PLY file.

    :param path: A path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
    'faces' (mx3 ndarray) - the latter three are optional.
    """
    f = open(path, 'r')

    n_pts = 0
    n_faces = 0
    face_n_corners = 3 # Only triangular faces are supported
    pt_props = []
    face_props = []
    text_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False

    # Read header
    while True:
        line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
        if line.startswith('element vertex'):
            n_pts = int(line.split(' ')[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith('element face'):
            n_faces = int(line.split(' ')[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith('element'): # Some other element
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split(' ')[-1], line.split(' ')[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split(' ')
            # (name of the property, data type)
            face_props.append(('n_corners', elems[2]))
            for i in range(face_n_corners):
                face_props.append(('ind_' + str(i), elems[3]))
        elif line.startswith('property2 list') and header_face_section:
            elems = line.split(' ')
            # (name of the property, data type)
            text_props.append(('n_corners', elems[2]))
            for i in range(3):
                text_props.append(('ind_' + str(i), elems[3]))
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures
    model = {}
    model['pts'] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

    pt_props_names = [p[0] for p in pt_props]
    is_normal = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), np.float)

    formats = { # For binary format
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

    # Load faces
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        print ('Error: Only triangular faces are supported.')
                        print ('Number of face corners:', val)
                        exit(-1)
                else:
                    prop_vals[prop[0]] = val
        else:

            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')

            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        print ('Error: Only triangular faces are supported.')
                        print ('Number of face corners:', int(elems[prop_id]))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = elems[prop_id]
        #print(prop_vals.keys())
        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])

    f.close()

    return model

def load_ply_my(path,bp):
    """
    Loads a 3D mesh model from a PLY file.

    :param path: A path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
    'faces' (mx3 ndarray) - the latter three are optional.
    """
    f = open(path, 'r')

    n_pts = 0
    n_faces = 0
    face_n_corners = 3 # Only triangular faces are supported
    pt_props = []
    face_props = []
    text_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False
    img = []
    W = 0
    H = 0
    # Read header
    while True:
        line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
        if line.startswith('element vertex'):
            n_pts = int(line.split(' ')[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith('element face'):
            n_faces = int(line.split(' ')[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith('element'): # Some other element
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('comment TextureFile'):  # Some other element
            img = cv2.imread(bp+'texture_map.png')
            img = cv2.flip(img,0)
            W = img.shape[1]
            H = img.shape[0]
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split(' ')[-1], line.split(' ')[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split(' ')
            # (name of the property, data type)
            face_props.append(('n_corners', elems[2]))
            for i in range(face_n_corners):
                face_props.append(('ind_' + str(i), elems[3]))
        elif line.startswith('property2 list') and header_face_section:
            elems = line.split(' ')
            # (name of the property, data type)
            text_props.append(('n_corners', elems[2]))
            for i in range(6):
                text_props.append(('ind_' + str(i), elems[3]))
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures
    model = {}
    model['pts'] = np.zeros((n_pts, 3), np.float)
    model['colors'] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

    pt_props_names = [p[0] for p in pt_props]
    is_normal = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), np.float)

    formats = { # For binary format
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

    # Load faces
    for face_id in range(n_faces):
        prop_vals = {}
        test_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        print ('Error: Only triangular faces are supported.')
                        print ('Number of face corners:', val)
                        exit(-1)
                else:
                    prop_vals[prop[0]] = val
        else:

            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')

            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        print ('Error: Only triangular faces are supported.')
                        print ('Number of face corners:', int(elems[prop_id]))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = elems[prop_id]
            if len(elems)>4:
                elems2 = elems[4:-1]
                for prop_id, prop in enumerate(text_props):
                    if prop[0] == 'n_corners':
                        if int(elems2[prop_id]) != 6:
                            print ('Error: Only triangular faces are supported.')
                            print ('Number of face corners:', int(elems2[prop_id]))
                            exit(-1)
                    else:
                        test_vals[prop[0]] = elems2[prop_id]


                r1,b1,g1,r2,b2,g2,r3,g3,b3 = coors2color(img, float(test_vals['ind_0']), float(test_vals['ind_1']), float(test_vals['ind_2']), float(test_vals['ind_3']), float(test_vals['ind_4']), float(test_vals['ind_5']), W, H)

                model['colors'][int(prop_vals['ind_0']),0] = r1
                model['colors'][int(prop_vals['ind_0']), 1] = g1
                model['colors'][int(prop_vals['ind_0']), 2] = b1

                model['colors'][int(prop_vals['ind_1']), 0] = r2
                model['colors'][int(prop_vals['ind_1']), 1] = g2
                model['colors'][int(prop_vals['ind_1']), 2] = b2

                model['colors'][int(prop_vals['ind_2']), 0] = r3
                model['colors'][int(prop_vals['ind_2']), 1] = g3
                model['colors'][int(prop_vals['ind_2']), 2] = b3
        #print(prop_vals.keys())
        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])


    f.close()

    return model

def coors2color(img, x1,y1,x2,y2,x3,y3,W,H):
# def coors2color(img, x1, x2, x3, y1, y2, y3, W, H):
#     img = cv2.flip(img,0)

    x1 = int(x1 * W)
    y1 = int(y1 * H)
    x2 = int(x2 * W)
    y2 = int(y2 * H)
    x3 = int(x3 * W)
    y3 = int(y3 * H)

    r1 = img[y1, x1, 0]
    r2 = img[y2, x2, 0]
    r3 = img[y3, x3, 0]

    g1 = img[y1, x1, 1]
    g2 = img[y2, x2, 1]
    g3 = img[y3, x3, 1]

    b1 = img[y1, x1, 2]
    b2 = img[y2, x2, 2]
    b3 = img[y3, x3, 2]
    #
    # r1 = img[x1, y1, 0]
    # r2 = img[x2, y2, 0]
    # r3 = img[x3, y3, 0]
    #
    # g1 = img[x1, y1, 1]
    # g2 = img[x2, y2, 1]
    # g3 = img[x3, y3, 1]
    #
    # b1 = img[x1, y1, 2]
    # b2 = img[x2, y2, 2]
    # b3 = img[x3, y3, 2]

    return r1,b1,g1,r2,b2,g2,r3,g3,b3



def save_ply(path, pts, pts_colors=np.array([]), faces=np.array([])):
    """
    Saves a 3D mesh model to a PLY file.

    :param path: A path to the resulting PLY file.
    :param pts: nx3 ndarray
    :param pts_colors;: nx3 ndarray
    :param faces: mx3 ndarray
    """
    pts_colors = np.array(pts_colors)
    if pts_colors.size != 0:
        assert(len(pts) == len(pts_colors))

    valid_pts_count = 0
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            valid_pts_count += 1

    f = open(path, 'w')
    f.write(
        'ply\n'
        'format ascii 1.0\n'
        #'format binary_little_endian 1.0\n'
        'element vertex ' + str(valid_pts_count) + '\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
    )
    if pts_colors.size != 0:
        f.write(
            'property uchar red\n'
            'property uchar green\n'
            'property uchar blue\n'
        )
    if faces.size != 0:
        f.write(
            'element face ' + str(len(faces)) + '\n'
            'property list uchar int vertex_indices\n'
        )
    f.write('end_header\n')

    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            f.write(' '.join(map(str, pt.squeeze().tolist())) + ' ')
            if pts_colors.size != 0:
                f.write(' '.join(map(str, map(int, list(pts_colors[pt_id])))))
            f.write('\n')
    for face in faces:
        f.write(' '.join(map(str, map(int, [len(face)] + list(face.squeeze())))) + ' ')
        f.write('\n')
    f.close()


def save_ply_with_color(path, pts, pts_colors=np.array([]), faces=np.array([])):
    """
    Saves a 3D mesh model to a PLY file.

    :param path: A path to the resulting PLY file.
    :param pts: nx3 ndarray
    :param pts_colors;: nx3 ndarray
    :param faces: mx3 ndarray
    """
    pts_colors = np.array(pts_colors)
    if pts_colors.size != 0:
        assert(len(pts) == len(pts_colors))

    valid_pts_count = 0
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            valid_pts_count += 1

    f = open(path, 'w')
    f.write(
        'ply\n'
        'format ascii 1.0\n'
        #'format binary_little_endian 1.0\n'
        'element vertex ' + str(valid_pts_count) + '\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
    )
    if pts_colors.size != 0:
        f.write(
            'property uchar red\n'
            'property uchar green\n'
            'property uchar blue\n'
        )
    if faces.size != 0:
        f.write(
            'element face ' + str(len(faces)) + '\n'
            'property list uchar int vertex_indices\n'
        )
    f.write('end_header\n')

    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            f.write(' '.join(map(str, pt.squeeze().tolist())) + ' ')
            if pts_colors.size != 0:
                f.write(' '.join(map(str, map(int, list(pts_colors[pt_id])))))
            f.write('\n')
    for face in faces:
        f.write(' '.join(map(str, map(int, [len(face)] + list(face.squeeze())))) + ' ')
        f.write('\n')
    f.close()