import numpy as np
from pathlib import Path as P
import pickle
import cv2
from vod.visualization.settings import label_color_palette_2d

import numpy as np
from pcdet.utils import calibration_kitti, object3d_kitti
from tqdm import tqdm
from vod.visualization.settings import label_color_palette_2d

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def transform_anno(loc, frame_id, is_radar=True, is_test=False):
    x, y, z = loc[0], loc[1], loc[2]
    modality = 'radar' if is_radar else 'lidar'
    split = 'testing' if is_test else 'training'

    calib_path = "/root/dj/code/CenterPoint-KITTI/data/vod_%s/%s/calib/%s.txt"%(modality, split, frame_id)

    # if is_radar:
    #     calib_path = "/root/dj/code/CenterPoint-KITTI/data/vod_radar/training/calib/{0}.txt".format(frame_id)
    # else:
    #     calib_path = "/root/dj/code/CenterPoint-KITTI/data/vod_lidar/training/calib/{0}.txt".format(frame_id)
    calib = calibration_kitti.Calibration(calib_path)
    loc = np.array([[x,y,z]])
    loc_lidar = calib.rect_to_lidar(loc)
    x,y,z = loc_lidar[0]
    return x,y,z


def get_rot_corner(x,y,l,w,a):

    s,c = np.sin(a),np.cos(a)

    corner_x = x - l/2
    corner_y = y - w/2

    corner_x -= x
    corner_y -= y

    new_corner_x = corner_x*c - corner_y*s 
    new_corner_y = corner_x*s + corner_y*c

    new_corner_x += x
    new_corner_y += y

    return new_corner_x,new_corner_y




def anno2plt(anno, color_dict, lw, frame_id, xz=False, is_radar=True, is_test=False):
    dim = anno['dimensions']
    loc = anno['location']
    # angle = anno['rotation_y'] * 180 / 3.14
    angle = -(anno['rotation_y']+ np.pi / 2) 
    rec_list = []
    cls = anno['name']
    for idx in range(dim.shape[0]):
        name = cls[idx]
        # print(name)
        if name not in color_dict:
            color = 'gray'
        else:
            color = color_dict[name]
            # print(color)
    
        if xz:

            x, _, y = transform_anno(loc[idx], frame_id, is_radar=is_radar, is_test=is_test)
            # w, _, l = dim[idx]
            l, w, _ = dim[idx]  # 
            ang = -angle[idx]* 0
        else:
            # print(loc[idx])
            x, y, z = transform_anno(loc[idx], frame_id, is_radar=is_radar, is_test=is_test)
            # print(x,y)
            
            ### X -> LENGTH
            ### Y -> WIDTH 
            ### Z -> HEIGHT, not used. 
            # x,y,z = loc[idx]
            # w, l, _ = dim[idx]
            # print(dim[idx]) 
            l, h, w  = dim[idx] # <-- SHOULD BE CORRECT? 
            ang = angle[idx]
            # ang = 0
            # print(l,w,ang)
            # print("="*40)

            ax,ay = get_rot_corner(x,y,l,w,ang)
            ang = ang * 180 / 3.14
            # ax = x - (l/4)
            # ay = y - (w/4)

        rec_list += [Rec((ax, ay), l, w, ang, fill=False, color=color,lw=lw)]
    return rec_list



def drawBEV(ax, pts, centers, annos, color_dict, frame_id, ax_title, ext_legends=[], is_radar=True, is_test=False):


    # 3. draw bbx
    try:
        rec_list = anno2plt(annos, color_dict, 2, frame_id=frame_id, xz=False, is_radar=is_radar, is_test=is_test)
    except:
        rec_list = anno2plt(annos[0], color_dict, 2, frame_id=frame_id, xz=False, is_radar=is_radar, is_test=is_test)
    
    # 1. draw original points if exist
    if pts is not None:
        x = pts[:, 1]
        y = pts[:, 2]
        ax.scatter(x, y, c='black', s=0.1)
    # 2. overlay centers
    if centers is not None:
        cx = centers[:, 1]
        cy = centers[:, 2]
        ax.scatter(cx, cy, c='red', s=0.1)
    
    for rec in rec_list:
        ax.add_patch(rec)

    legend_elements = [Patch(facecolor='white', edgecolor=v, label=k) for i, (k, v) in enumerate(color_dict.items())]
    if centers is not None:
        legend_elements += [Line2D([0], [0], marker='o', color='w', label='FG points',
                          markerfacecolor='r', markersize=10)]
    legend_elements += ext_legends
    ax.legend(handles=legend_elements, loc=1)
    ax.set_title(ax_title)


def draw_rectangle(ax, anno, color_dict, xz=False):
    recs = anno2plt(anno, color_dict, lw=2, xz=xz)
    for rec in recs:
        ax.add_patch(rec)

def saveODImgs(frame_ids, anno, data_path, img_path, color_dict, is_radar=True, title='pred', limit_range=None, is_test=False):
    print('=================== drawing images ===================')
    plt.rcParams['figure.dpi'] = 150
    for fid in tqdm(frame_ids):
        pcd_fname = data_path / (fid + '.bin')
        vis_pcd = get_radar(pcd_fname) if is_radar else get_lidar(pcd_fname, limit_range=limit_range)
        vis_pcd = pcd_formating(vis_pcd)
        ax = plt.gca()
        drawBEV(ax, vis_pcd, None, anno[fid], color_dict, fid, title, is_radar=is_radar, is_test=is_test)
        plt.xlim(-0,75)
        plt.ylim(-30,30)
        img_fname = img_path / (fid + '.png')
        plt.savefig(str(img_fname))
        plt.cla()

def saveODImgs_multithread(frame_ids, anno, data_path, img_path, color_dict, is_radar=True, title='pred', limit_range=None, is_test=False, workers=8):
    print('=================== drawing images ===================')
    plt.rcParams['figure.dpi'] = 150
    from tqdm.contrib.concurrent import process_map

    def draw_img(fid):
        pcd_fname = data_path / (fid + '.bin')
        vis_pcd = get_radar(pcd_fname) if is_radar else get_lidar(pcd_fname, limit_range=limit_range)
        vis_pcd = pcd_formating(vis_pcd)
        ax = plt.gca()
        drawBEV(ax, vis_pcd, None, anno[fid], color_dict, fid, title, is_radar=is_radar, is_test=is_test)
        plt.xlim(-0,75)
        plt.ylim(-30,30)
        img_fname = img_path / (fid + '.png')
        plt.savefig(str(img_fname))
        plt.cla()
    def print_fid(fids):
        print(fids)
    process_map(print_fid, frame_ids, max_workers=workers, chunksize=len(frame_ids)//workers)

def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
            & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \
            & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])
    return mask

def get_radar(fname):
    assert fname.exists()
    radar_point_cloud = np.fromfile(str(fname), dtype=np.float32).reshape(-1, 7)
    return radar_point_cloud

def get_lidar(fname, limit_range):
    assert fname.exists()
    lidar_point_cloud = np.fromfile(str(fname), dtype=np.float32).reshape(-1, 4)
    if limit_range is not None:
        mask = mask_points_by_range(lidar_point_cloud, limit_range)
        return lidar_point_cloud[mask]
    else:
        return lidar_point_cloud

def pcd_formating(pcd):
    num_pts = pcd.shape[0]
    zeros_pad = np.zeros([num_pts, 1])
    final_pcd = np.concatenate((zeros_pad, pcd), axis=1)
    return final_pcd

def make_vid(imgs, vid_fname, fps=15):
    print('=================== making videos ===================')
    out = None
    for fname in tqdm(imgs):
        i = cv2.imread(fname)
        if out is None:
            h, w, _ = i.shape
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(vid_fname), fourcc, fps, size)
            
        out.write(i)
    out.release()

def get_label(label_file):
    return object3d_kitti.get_objects_from_label(label_file)

def load_result_file(label_file):
    obj_list = get_label(label_file)
    annotations = {}
    if len(obj_list) == 0:
        annotations['name'] = np.array([])       
        annotations['truncated'] = np.array([])
        annotations['occluded'] = np.array([])
        annotations['alpha'] = np.array([])
        annotations['bbox'] = np.array([])
        annotations['dimensions'] = np.array([])
        annotations['location'] = np.array([])
        annotations['rotation_y'] = np.array([])
        annotations['score'] = np.array([])
        annotations['difficulty'] = np.array([])

    else:
        annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
        annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
        annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
        annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
        annotations['score'] = np.array([obj.score for obj in obj_list])
        annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

    return annotations
