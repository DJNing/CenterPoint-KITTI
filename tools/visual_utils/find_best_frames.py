import io as sysio
import torch
import numba
import numpy as np
import pickle
from pathlib import Path as P
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.kitti.kitti_object_eval_python.eval import clean_data,_prepare_data,eval_class,get_mAP,get_mAP_R40
from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import get_label_annos
from vod.visualization.settings import label_color_palette_2d
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle 
from matplotlib.patches import Rectangle as Rec
import numpy as np
from tqdm import tqdm
from vod import frame
from vod.visualization.settings import label_color_palette_2d
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameLabels, FrameTransformMatrix
from vod.frame.transformations import transform_pcl
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from vod.visualization import Visualization3D
from skimage import io
from vis_tools import fov_filtering, make_vid
from glob import glob
from collections import Counter
import matplotlib.cm as cm
import os
# from vis_tools import fov_filtering
from val_path_dict import path_dict, det_path_dict
import shutil
from draw_box_count_mAP import count_points_in_box



def eval_individual_frames(gt_annos,dt_annos,box_count_threshold=None):

    results = []
    frame_ids = []
    
    for i in tqdm(range(len(gt_annos))):

        GT = []
        DT = []
        GT += [gt_annos[i]]
        DT += [dt_annos[i]]
        
        frame_ids += [dt_annos[i]['frame_id']]
        current_classes = [0,1,2]
        difficulties = [0]
        overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
        iou_threshold = np.expand_dims(overlap_0_7,axis=0)

        ret = eval_class(GT, DT, current_classes, difficulties, 2,
                        iou_threshold,box_count_threshold=box_count_threshold)
        mAP_3d = np.round(get_mAP(ret["precision"]),4)
        mAP_3d_R40 = np.round(get_mAP_R40(ret["precision"]),4)
        results += [mAP_3d]

    all_classes_AP = np.stack(results).squeeze(2).squeeze(2)
    AP_sum = np.sum(all_classes_AP,axis=1)
    mAP = torch.from_numpy(AP_sum/3)
    return all_classes_AP,mAP, np.array(frame_ids)




def get_top_frames_point_count(gt,dt,counts,k):
    all_classes_AP,mAP,frame_ids = eval_individual_frames(gt,dt,box_count_threshold=counts[0])
    print('')





def get_top_frames_mAP(new_gt,new_dt,k):
    classes_AP,mAP,frames = eval_individual_frames(new_gt,new_dt)
    top_frame_mAP, top_idx = torch.topk(mAP,k)
    top_frame_ids = frames[top_idx]
    
    print(f'top {k} mAPs')
    print(top_frame_mAP)
    print('='*50)
    print(f'top {k} frame_ids')
    print(top_frame_ids)

    return top_frame_ids



def get_top_frames_counts(gt,dt,k):
    name_to_int = {'Car':0,'Pedestrian':1,'Cyclist':2}
    N = len(dt)
    dt_counter = torch.zeros((N,3))
    gt_counter = torch.zeros((N,3))
    frame_counter = torch.zeros((N))
    for i in range(len(gt)):
        preds = [c for c in dt[i]['name']]
        pred_counts = Counter(preds)
        for c in pred_counts:
            dt_counter[i][name_to_int[c]] = pred_counts[c]

        gt_labels = [c for c in gt[i]['name'] if c in ['Car','Pedestrian','Cyclist']]
        label_counts = Counter(gt_labels)
        for c in label_counts:
            gt_counter[i][name_to_int[c]] = label_counts[c]
        
        frame_counter[i] = int(dt[i]['frame_id'])    

    class_diff = torch.abs(gt_counter-dt_counter)
    total_diff = torch.sum(class_diff,axis=1)

    valid_scenes = torch.nonzero(torch.sum(gt_counter,axis=1))
    frame_ids = frame_counter.type(torch.int32)
    
    valid_diff = total_diff[valid_scenes].squeeze()
    valid_frames = frame_ids[valid_scenes].squeeze()


    best_frames = torch.topk(valid_diff,k,largest=False)
    v, idx = best_frames

    best_frame_ids = valid_frames[idx]
    print(f'top {k} diff')
    print(v)
    print('='*50)
    print(f'top {k} frame_ids')
    print(best_frame_ids)
    return best_frame_ids



def softlink_baselines(
    frame_ids,
    main_tag,
    baseline_tags,
    file_name,
    method,
    is_lidar,
    do_softlink=False):

    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    root_dir = base_path /'output' / 'vod_vis' / 'det_comparisons' / file_name / method 

    all_tags = [main_tag,'lidar_GT'] + baseline_tags if is_lidar else [main_tag,'radar_GT'] + baseline_tags 

    for frame in tqdm(frame_ids):
        write_path = root_dir / str(frame).zfill(5)
        os.makedirs(write_path,exist_ok=True)

        for tag in all_tags:
            img_path = base_path / (det_path_dict[tag] + f"/{str(frame).zfill(5)}.png")
           
            out_path = write_path / f"{tag}.png"

            if do_softlink:
                os.symlink(img_path,out_path)
            else:
                shutil.copy(str(img_path),out_path)




def load_gt_dt(tag,count_points=False,is_radar=False):
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    result_path = base_path / path_dict[tag]

    # loading detections and GT
    with open(str(result_path / 'gt.pkl'), 'rb') as f:
        gt = pickle.load(f)

    with open(str(result_path / 'dt.pkl'), 'rb') as f:
        dt = pickle.load(f)

    if count_points:
        modality = 'radar' if  is_radar else 'lidar'
        data_path = base_path / ('data/vod_%s/training/velodyne'%modality )
        
        gt = count_points_in_box(gt, is_radar, is_dt=False,data_path=data_path)
        dt = count_points_in_box(dt, is_radar, is_dt=True,data_path=data_path)




    new_gt = [gt[key] for key in gt.keys()]
    new_dt = [dt[key][0] for key in dt.keys()]

    return new_gt,new_dt
def get_lidar_range():
    # start_range = [0, 1, 20, 60, 120, 200]
    # end_range = [1, 20, 60, 120, 200, float('inf')]
    start_range = [0]
    end_range = [40]
    result = np.array([start_range, end_range])
    return result.T

def main():
    '''
    RETURNS BEST FRAMES FOR CHOSEN TAG BASED ON TWO METHODS
    top_frames_mAP: for each frame, calculate the mAP, then returns topk frames 
    top_frames_counts: for each frame, count the difference of predicted obj vs label, return mink frames
        example: GT = 3 cars, 2 ped, 1 cyclist
                 DT = 2 cars, 1 ped, 0 cyclist
               diff = 1 + 1 + 1 = 3
                "best" frames are the ones where diff is closest to 0                
    '''    
    # ! SETTINGS
    tag = 'CFAR_lidar_rcs'
    k = 30 # top k frames
    baseline_tags = ['']
    # ! SETTINGS 

    print(f'Evaluating {tag} with path {path_dict[tag]}')
    


    # gt,dt = load_gt_dt(tag,count_points=True,is_radar=False)


    # top_frame_ids_mAP = get_top_frames_mAP(gt,dt,k)
    # top_frame_ids_counts = get_top_frames_counts(gt,dt,k).numpy()
    # counts = get_lidar_range()
    # get_top_frames_point_count(gt,dt,counts,k)



    baseline_tags = [
    'lidar_i',
    'second_lidar',
    'pp_lidar',
    '3dssd_lidar',
    'centerpoint_lidar',
    'pvrcnn_lidar',
    'pointrcnn_lidar'
    ]

    # frame_seq = [ str(v).zfill(5) for v in list(range(110,171))]

    frame_seq = ['00154', '00150', '00213', '00158', '00195', '00157', '00151',
       '00217', '00171', '00196', '00160', '00178', '00164', '00155',
       '00214', '00211', '00224', '00220', '00208', '00190']
    softlink_baselines(
        frame_seq,
        main_tag=tag,
        baseline_tags=baseline_tags,
        file_name='best_ped',
        method='mAP',
        is_lidar=True)

    # softlink_baselines(
    #     top_frame_ids_counts,
    #     main_tag=tag,
    #     baseline_tags=baseline_tags,
    #     method='count',
    #     is_lidar=True)



if __name__ == "__main__":
    main()





# def eval_specific_frame(gt_annos,dt_annos,idx):
#     results = []
#     frame_ids = []
    
    

#     GT = []
#     DT = []
#     GT += [gt_annos[idx]]
#     DT += [dt_annos[idx]]
    
#     frame_ids += [dt_annos[idx]['frame_id']]
#     current_classes = [0,1,2]
#     difficulties = [0]
#     overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
#                             [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
#                             [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
#     iou_threshold = np.expand_dims(overlap_0_7,axis=0)

#     ret = eval_class(GT, DT, current_classes, difficulties, 2,
#                     iou_threshold)
#     mAP_3d = np.round(get_mAP(ret["precision"]),4)
#     mAP_3d_R40 = np.round(get_mAP_R40(ret["precision"]),4)
#     results += [mAP_3d]

#     all_classes_AP = np.stack(results).squeeze(2).squeeze(2)
#     AP_sum = np.sum(all_classes_AP,axis=1)
#     mAP = torch.from_numpy(AP_sum/3)
#     return all_classes_AP,mAP, np.array(frame_ids)


# def eval_all(gt_annos,dt_annos):

#     current_classes = [0,1,2]
#     difficulties = [0]
#     overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
#                             [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
#                             [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
#     iou_threshold = np.expand_dims(overlap_0_7,axis=0)

#     ret = eval_class(gt_annos, dt_annos, current_classes, difficulties, 2,
#                     iou_threshold)
#     mAP_3d = np.round(get_mAP(ret["precision"]),4)
#     mAP_3d_R40 = np.round(get_mAP_R40(ret["precision"]),4)
#     return mAP_3d