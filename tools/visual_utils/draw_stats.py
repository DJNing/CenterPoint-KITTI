import io as sysio

import numba
import numpy as np
import pickle
from pathlib import Path as P
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.kitti.kitti_object_eval_python.eval import clean_data,_prepare_data,eval_class,get_mAP,get_mAP_R40
from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import get_label_annos
from vod.visualization.settings import label_color_palette_2d
import matplotlib.pyplot as plt


def draw_iou_results(iou_thresholds,
                car_AP,
                pedestrian_AP,
                cyclist_AP,
                result_dir,
                is_distance=False,
                fig_name=None,
                xlabel=None):
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

    fig.savefig(fig_path)
    


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
    distances = [[0,10],
            [10,20],
            [20,30],
            [30,40],
            [40,50],
    ]
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    
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





test_path = 'output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/eval/best_epoch_checkpoint'

abs_path = P(__file__).parent.resolve()
base_path = abs_path.parents[1]
result_path = base_path / test_path


current_classes = [0,1,2]

with open(str(result_path / 'gt.pkl'), 'rb') as f:
    gt = pickle.load(f)

# load det
with open(str(result_path / 'dt.pkl'), 'rb') as f:
    dt = pickle.load(f)

new_gt = []
for key in gt.keys():
    temp_dict = {}
    new_gt += [gt[key]] 

new_dt = []
for key in dt.keys():
    temp_dict = {}
    new_dt += [dt[key][0]] 



all_iou_results = get_all_iou_results(new_gt,new_dt,current_classes)


draw_iou_results(all_iou_results['iou_thresholds'],
                all_iou_results['Car']['mAP_3d'],
                all_iou_results['Pedestrian']['mAP_3d'],
                all_iou_results['Cyclist']['mAP_3d'],
                result_path)


distance_results = get_results_over_distance(new_gt,new_dt,current_classes)
                
draw_iou_results(distance_results['distances'],
                distance_results['Car']['mAP_3d'],
                distance_results['Pedestrian']['mAP_3d'],
                distance_results['Cyclist']['mAP_3d'],
                result_path)



