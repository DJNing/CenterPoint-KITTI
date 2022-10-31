import io as sysio
import torch
import numpy as np
import pickle
from pathlib import Path as P
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.kitti.kitti_object_eval_python.eval import clean_data,_prepare_data,eval_class,get_mAP,get_mAP_R40
from tqdm import tqdm
import pickle 
import numpy as np
from tqdm import tqdm
from collections import Counter
import os
from val_path_dict import path_dict, det_path_dict,lidar_baseline_tags
import shutil
from draw_box_count_mAP import count_points_in_box
from draw_3d import do_vis
from functools import reduce


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




def get_top_frames_point_count(gt,dt,k):
    counts = get_lidar_range()
    all_classes_AP,mAP,frame_ids = eval_individual_frames(gt,dt,box_count_threshold=counts[0])
    all_classes_tens = torch.from_numpy(all_classes_AP)
    ped_AP = all_classes_tens[:,1]
    v, i = torch.topk(ped_AP,k)

    print(frame_ids[i])
    return frame_ids[i]
    # print('')





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


    WORKFLOW:
    1. RUN THIS SCRIPT, this saves a npy file with frame #s to generate
    2a. run 'source py3env/bin/activate' and 'export PYTHONPATH="${PYTHONPATH}:/root/gabriel/code/parent/CenterPoint-KITTI"'
    2b. in the same terminal window run draw3d.generate_comparison_frames(), this creates the images
    3. comeback and run the softlnk_baselines()

    # TODO: Just generate all the frames so step 1 can be skipped



    '''    
    # ! SETTINGS #####################
    tag = 'CFAR_lidar_rcs'
    k = 30 # top k frames
    resolution = '480'
    is_test_set = False
    vod_data_path = '/mnt/12T/public/view_of_delft'

    # ! SETTINGS #####################

    print(f'Evaluating {tag} with path {path_dict[tag]}')
    


    
    gt,dt = load_gt_dt(tag,count_points=True,is_radar=False)
    top_frame_ids_mAP = get_top_frames_mAP(gt,dt,k)
    top_frame_ids_counts = get_top_frames_counts(gt,dt,k).numpy()
    top_frame_points = get_top_frames_point_count(gt,dt,k)
    
    ret_dict = {
        'mAP': top_frame_ids_mAP,
        'counts': top_frame_ids_counts,
        'box_point_count': top_frame_points
    }
    frames_to_generate = reduce(np.union1d, (top_frame_ids_mAP,top_frame_ids_counts,top_frame_points))
    
    print('='*30,'frames to generate','='*30)
    print(frames_to_generate)
    with open('frames_to_generate.npy', 'wb') as f:
        np.save(f,frames_to_generate)

    for t in [tag]+lidar_baseline_tags:
        do_vis(t,frames_to_generate,is_test_set,None,vod_data_path,resolution)


    # for k in ret_dict.keys():    
    #     softlink_baselines(
    #         ret_dict[k],
    #         main_tag=tag,
    #         baseline_tags=baseline_tags,
    #         method=k,
    #         file_name='test',
    #         is_lidar=True)

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