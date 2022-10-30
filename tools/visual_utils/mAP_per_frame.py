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

def print_results(all_iou_results,distance_results):
    print('*************   AP for different IoU thresholds   *************')
    iou_threshold = all_iou_results['iou_thresholds']
    car_APs = all_iou_results['Car']['mAP_3d']
    pedestrian_APs = all_iou_results['Pedestrian']['mAP_3d']
    cyclist_APs = all_iou_results['Cyclist']['mAP_3d']

    print(f"IoU thresholds: {iou_threshold}")
    print(f"Car APs: {car_APs}")
    print(f"Pedestrian APs: {pedestrian_APs}")
    print(f"Cyclist APs: {cyclist_APs}")

    print('*************   AP(0.5,0.25,0.25) at different distances    *************')
    distances = distance_results['distances']
    car_APs = distance_results['Car']['mAP_3d']
    pedestrian_APs = distance_results['Pedestrian']['mAP_3d']
    cyclist_APs = distance_results['Cyclist']['mAP_3d']

    print(f"Distance Ranges: {distances}")
    print(f"Car APs: {car_APs}")
    print(f"Pedestrian APs: {pedestrian_APs}")
    print(f"Cyclist APs: {cyclist_APs}")

def draw_iou_results(iou_thresholds,
                car_AP,
                pedestrian_AP,
                cyclist_AP,
                result_dir,
                is_distance=False,
                fig_name=None,
                xlabel=None,
                fig_title=None):
    fig, ax = plt.subplots(1)

    car_color = label_color_palette_2d['Car']
    ped_color = label_color_palette_2d['Pedestrian']
    cyclist_color = label_color_palette_2d['Cyclist']

    mAP = np.mean([car_AP,pedestrian_AP,cyclist_AP],axis=0)
    
    ax.scatter(iou_thresholds,car_AP,color=car_color,clip_on=False)
    ax.scatter(iou_thresholds,pedestrian_AP,color=ped_color,clip_on=False)
    ax.scatter(iou_thresholds,cyclist_AP,color=cyclist_color,clip_on=False)
    ax.scatter(iou_thresholds,mAP,color='black',clip_on=False)
    
    ax.plot(iou_thresholds,car_AP,color=car_color,label='Car')
    ax.plot(iou_thresholds,pedestrian_AP,color=ped_color,label='Pedestrian')
    ax.plot(iou_thresholds,cyclist_AP,color=cyclist_color,label='Cyclist')
    ax.plot(iou_thresholds,mAP,color='black',label='mAP')

    if xlabel is not None:
        ax.set_xlabel(xlabel) 
    else:
        ax.set_xlabel('IoU threshold (3D)') 
    ax.set_ylabel('AP (3D IoU)')

    ax.set_yticks(np.arange(0,110,10))
    
    if not is_distance:
        ax.set_xticks(np.arange(0,1,0.1))
    
    ax.grid(axis = 'y')
    
    ax.legend()

    for label in ax.get_yticklabels()[1::2]:
        label.set_visible(False)
        plt.xlim(xmin=0) 

    plt.ylim(ymin=0,ymax=100)

    if fig_name is not None:
        fig_path = result_dir / (fig_name+'.png')
    else:
        fig_path = result_dir / 'iou_threshold.png'

    if fig_title is not None:
        ax.set_title(fig_title)

    fig.savefig(fig_path)
    plt.close()


def get_all_iou_results(gt_annos,
            dt_annos,
            current_classes,
            compute_aos=False,
            PR_detail_dict=None):
    difficulties = [0]

    iou_threshold = np.expand_dims(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]),axis=0)
    increment = np.expand_dims(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]),axis=0)
    starting_iou = 0.1
    category_results = {
        'Car': {
            'mAP_3d': [],
            'mAP_3d_R40': []
        },
        'Pedestrian':{
            'mAP_3d': [],
            'mAP_3d_R40': []
        },
        'Cyclist':{
            'mAP_3d': [],
            'mAP_3d_R40': []
        },
        'iou_thresholds': np.round(np.arange(0.1,1,0.1),2)
    }

    for i in range(9):                
        ret = eval_class(gt_annos, dt_annos, current_classes, difficulties, 2,
                     iou_threshold)
        mAP_3d = np.round(get_mAP(ret["precision"]),4)
        mAP_3d_R40 = np.round(get_mAP_R40(ret["precision"]),4)

        category_results["Car"]['mAP_3d'] += [mAP_3d[0].item()]
        category_results["Car"]['mAP_3d_R40'] += [mAP_3d_R40[0].item()]
                
        category_results["Pedestrian"]['mAP_3d'] += [mAP_3d[1].item()]
        category_results["Pedestrian"]['mAP_3d_R40'] += [mAP_3d_R40[1].item()]
        
        category_results["Cyclist"]['mAP_3d'] += [mAP_3d[2].item()]
        category_results["Cyclist"]['mAP_3d_R40'] += [mAP_3d_R40[2].item()]

        starting_iou += 0.1
        iou_threshold += increment

    return category_results

def get_results_over_distance(gt_annos,
            dt_annos,
            current_classes):
    difficulties = [0]
    distances = [[0,20],
            [20,40],
            [40,100],
            [0,100]
    ]
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])

    iou_threshold = np.expand_dims(overlap_0_5,axis=0)

    distance_results = {
        'Car': {
            'mAP_3d': [],
            'mAP_3d_R40': []
        },
        'Pedestrian':{
            'mAP_3d': [],
            'mAP_3d_R40': []
        },
        'Cyclist':{
            'mAP_3d': [],
            'mAP_3d_R40': []
        },
        'distances': [f"{d[0]}-{d[1]} m" for d in distances]
    }

    for d in distances:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficulties, 2,
                     iou_threshold,distance_range=d)

        mAP_3d = np.round(get_mAP(ret["precision"]),4)
        mAP_3d_R40 = np.round(get_mAP_R40(ret["precision"]),4)

        distance_results["Car"]['mAP_3d'] += [mAP_3d[0].item()]
        distance_results["Car"]['mAP_3d_R40'] += [mAP_3d_R40[0].item()]
                
        distance_results["Pedestrian"]['mAP_3d'] += [mAP_3d[1].item()]
        distance_results["Pedestrian"]['mAP_3d_R40'] += [mAP_3d_R40[1].item()]
        
        distance_results["Cyclist"]['mAP_3d'] += [mAP_3d[2].item()]
        distance_results["Cyclist"]['mAP_3d_R40'] += [mAP_3d_R40[2].item()]

    return distance_results






def eval_individual_frames(gt_annos,dt_annos):

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
                        iou_threshold)
        mAP_3d = np.round(get_mAP(ret["precision"]),4)
        mAP_3d_R40 = np.round(get_mAP_R40(ret["precision"]),4)
        results += [mAP_3d]

    all_classes_AP = np.stack(results).squeeze(2).squeeze(2)
    AP_sum = np.sum(all_classes_AP,axis=1)
    mAP = torch.from_numpy(AP_sum/3)


    return mAP, np.array(frame_ids)
    




def main():
    path_dict = {
        # 'CFAR_radar':'output/IA-SSD-GAN-vod-aug/radar48001_512all/eval/best_epoch_checkpoint',
        # 'radar_rcsv':'output/IA-SSD-vod-radar/iassd_best_aug_new/eval/best_epoch_checkpoint',
        # 'radar_rcs':'output/IA-SSD-vod-radar/iassd_rcs/eval/best_epoch_checkpoint',
        # 'radar_v':'output/IA-SSD-vod-radar/iassd_vcomp_only/eval/best_epoch_checkpoint',
        # # 'radar':'output/IA-SSD-vod-radar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        # # 'lidar':'output/IA-SSD-vod-lidar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        # 'CFAR_lidar_rcsv':'output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcs':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_rcs_only/eval/best_epoch_checkpoint',
        # 'CFAR_lidar_v':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_vcomp_only/eval/best_epoch_checkpoint',
        # 'CFAR_lidar':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_xyz_only/eval/best_epoch_checkpoint',
        'lidar_i':'output/IA-SSD-vod-lidar/all_cls/eval/checkpoint_epoch_80',
        # # 'pp_radar_rcs' : 'output/pointpillar_vod_radar/debug_new/eval/checkpoint_epoch_80',
        # 'pp_radar_rcsv' : 'output/pointpillar_vod_radar/vrcomp/eval/best_epoch_checkpoint', 
        # # '3dssd_radar_rcs': 'output/3DSSD_vod_radar/rcs/eval/best_epoch_checkpoint',
        # '3dssd_radar_rcsv': 'output/3DSSD_vod_radar/vcomp/eval/best_epoch_checkpoint',
        # # 'centerpoint_radar_rcs': 'output/centerpoint_vod_radar/rcs/eval/best_epoch_checkpoint',
        # 'centerpoint_radar_rcsv': 'output/centerpoint_vod_radar/rcsv/eval/best_epoch_checkpoint',
        'pp_lidar': 'output/pointpillar_vod_lidar/debug_new/eval/checkpoint_epoch_80',
        '3dssd_lidar': 'output/3DSSD_vod_lidar/all_cls/eval/checkpoint_epoch_80',
        'centerpoint_lidar': 'output/centerpoint_vod_lidar/xyzi/eval/best_epoch_checkpoint'
    }

    ALL_mAP = []
    processed_tags = []
    for tag in path_dict.keys():
        processed_tags += [tag]
        abs_path = P(__file__).parent.resolve()
        base_path = abs_path.parents[1]
        result_path = base_path / path_dict[tag]
        
        save_base_path = base_path /'output' / 'vod_vis' / 'stats_newrange_070505'
        save_base_path.mkdir(parents=True,exist_ok=True)
        
        # result_path = save_base_path
        print(f'*************   DRAWING PLOTS FOR TAG:{path_dict[tag]}   *************')

        with open(str(result_path / 'gt.pkl'), 'rb') as f:
            gt = pickle.load(f)

        with open(str(result_path / 'dt.pkl'), 'rb') as f:
            dt = pickle.load(f)

        # Car, ped, cyclis
        current_classes = [0,1,2]
        
        # load gt boxes
        new_gt = []
        for key in gt.keys():
            new_gt += [gt[key]] 

        # load predicted boxes 
        new_dt = []
        for key in dt.keys():
            new_dt += [dt[key][0]] 


        

        mAP,frames = eval_individual_frames(new_gt,new_dt)
        ALL_mAP += [mAP]

    k = 30

    f = open(f"top{k}.txt", "w")

    combined_mAP = torch.stack(ALL_mAP)
    cfar_mAP = combined_mAP[0,:]
    remaining_mAP = combined_mAP[1:,:]
    mAP_difference = cfar_mAP - remaining_mAP
    topk = torch.topk(mAP_difference,k)
    v,i = topk

    frames = np.expand_dims(frames,0)
    repeated_frames = np.repeat(frames,4,axis=0).astype(int)
    frames_tensor = torch.from_numpy(repeated_frames)

    top_frames = torch.gather(frames_tensor,1,i)


    print(processed_tags)
    print(top_frames)


        

       

if __name__ == "__main__":
    main()




