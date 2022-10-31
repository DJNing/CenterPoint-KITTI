# %%
# import torch
import matplotlib.pyplot as plt
import pickle 
from pathlib import Path as P
from matplotlib.patches import Rectangle as Rec
import numpy as np
from tqdm import tqdm
from vod import frame
# import sys
# sys.path.append("/Users/gabrielchan/Desktop/code/CenterPoint-KITTI")
# from pcdet.utils import calibration_kitti
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
from val_path_dict import path_dict,lidar_path_dict,lidar_baseline_tags



## import from visualization_2D instead
def get_pred_dict(dt_file):
    '''
    reads results.pkl file
    returns dictionary with str(frame_id) as key, and list of strings, 
    where each string is a predicted box in kitti format.
    '''
    dt_annos = []

    # load detection dict
    with open(dt_file, 'rb') as f:
        infos = pickle.load(f)
        dt_annos.extend(infos)      
    labels_dict = {}
    for j in range(len(dt_annos)):
        labels = []
        curr = dt_annos[j]
        frame_id = curr['frame_id']
        
        # no predicted 
        if len(dt_annos[j]['name']) == 0: 
            labels += []
        
        else:
            for i in range(len(dt_annos[j]['name'])):       
                # extract the relevant info and format it 
                line = [str(curr[x][i]) if not isinstance(curr[x][i],np.ndarray) else [y for y in curr[x][i]]  for x in list(curr.keys())[:-2]]
                flat = [str(num) for item in line for num in (item if isinstance(item, list) else (item,))]
                
                # L,H,W -> H,W,L 
                flat[9],flat[10] = flat[10],flat[9]
                flat[8],flat[10] = flat[10],flat[8]
                
                labels += [" ".join(flat)]

        labels_dict[frame_id] = labels
    return labels_dict


def vod_to_o3d(vod_bbx,vod_calib):
    # modality = 'radar' if is_radar else 'lidar'
    # split = 'testing' if is_test else 'training'    
    

    
    COLOR_PALETTE = {
        'Cyclist': (1, 0.0, 0.0),
        'Pedestrian': (0.0, 1, 0.0),
        'Car': (0.0, 0.3, 1.0),
        'Others': (0.75, 0.75, 0.75)
    }

    box_list = []
    for box in vod_bbx:
        if box['label_class'] in ['Cyclist','Pedestrian','Car']:
            # Conver to lidar_frame 
            # NOTE: O3d is stupid and plots the center of the box differently,
            offset = -(box['h']/2) 
            old_xyz = np.array([[box['x'],box['y']+offset,box['z']]])
            xyz = transform_pcl(old_xyz,vod_calib.t_lidar_camera)[0,:3] #convert frame
            extent = np.array([[box['l'],box['w'],box['h']]])
            
            # ROTATION MATRIX
            rot = -(box['rotation']+ np.pi / 2) 
            angle = np.array([0, 0, rot])
            rot_matrix = R.from_euler('XYZ', angle).as_matrix()
            
            # CREATE O3D OBJECT
            obbx = o3d.geometry.OrientedBoundingBox(xyz, rot_matrix, extent.T)
            obbx.color = COLOR_PALETTE.get(box['label_class'],COLOR_PALETTE['Others']) # COLOR
            
            box_list += [obbx]

    return box_list







def get_kitti_locations(vod_data_path):
    kitti_locations = KittiLocations(root_dir=vod_data_path,
                                output_dir="output/",
                                frame_set_path="",
                                pred_dir="",
                                )
    return kitti_locations
                             
def get_visualization_data_true_frame(kitti_locations,dt_path,frame_id,is_test_set):


    if is_test_set:
        frame_ids  = [P(f).stem for f in glob(str(dt_path)+"/*")]
        frame_data = FrameDataLoader(kitti_locations,
                                frame_id,"",dt_path)
        vod_calib = FrameTransformMatrix(frame_data)
        

    else:
        pred_dict = get_pred_dict(dt_path)
        frame_ids = list(pred_dict.keys())
        frame_data = FrameDataLoader(kitti_locations,
                                frame_id,pred_dict)
        vod_calib = FrameTransformMatrix(frame_data)

    # print(len(frame_ids))
    



    # get pcd
    original_radar = frame_data.radar_data
    radar_points = transform_pcl(original_radar,vod_calib.t_lidar_radar)
    radar_points,flag = fov_filtering(radar_points,frame_id,is_radar=False,return_flag=True)
    lidar_points = frame_data.lidar_data 
    lidar_points = fov_filtering(lidar_points,frame_id,is_radar=True)

    
    colors = cm.spring(original_radar[flag][:,4])[:,:3]



    # convert into o3d pointcloud object
    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = o3d.utility.Vector3dVector(radar_points[:,0:3])
    # radar_colors = np.ones_like(radar_points[:,0:3])
    radar_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    lidar_pcd = o3d.geometry.PointCloud()
    lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points[:,0:3])
    lidar_colors = np.ones_like(lidar_points[:,0:3])
    lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_colors)

    
    if is_test_set:
        vod_labels = None
        o3d_labels = None 
    else:
        vod_labels = FrameLabels(frame_data.get_labels()).labels_dict
        o3d_labels = vod_to_o3d(vod_labels,vod_calib)    

    vod_preds = FrameLabels(frame_data.get_predictions()).labels_dict
    o3d_predictions = vod_to_o3d(vod_preds,vod_calib)
    

    vis_dict = {
        'radar_pcd': [radar_pcd],
        'lidar_pcd': [lidar_pcd],
        'vod_predictions': vod_preds,
        'o3d_predictions': o3d_predictions,
        'vod_labels': vod_labels,
        'o3d_labels': o3d_labels,
        'frame_id': frame_id
    }
    return vis_dict

def get_visualization_data(kitti_locations,dt_path,frame_id,is_test_set):


    if is_test_set:
        frame_ids  = [P(f).stem for f in glob(str(dt_path)+"/*")]
        frame_data = FrameDataLoader(kitti_locations,
                                frame_ids[frame_id],"",dt_path)
        vod_calib = FrameTransformMatrix(frame_data)
        

    else:
        pred_dict = get_pred_dict(dt_path)
        frame_ids = list(pred_dict.keys())
        frame_data = FrameDataLoader(kitti_locations,
                                frame_ids[frame_id],pred_dict)
        vod_calib = FrameTransformMatrix(frame_data)

    # print(len(frame_ids))
    



    # get pcd
    original_radar = frame_data.radar_data
    radar_points = transform_pcl(original_radar,vod_calib.t_lidar_radar)
    radar_points,flag = fov_filtering(radar_points,frame_ids[frame_id],is_radar=False,return_flag=True)
    lidar_points = frame_data.lidar_data 
    lidar_points = fov_filtering(lidar_points,frame_ids[frame_id],is_radar=True)

    
    colors = cm.spring(original_radar[flag][:,4])[:,:3]



    # convert into o3d pointcloud object
    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = o3d.utility.Vector3dVector(radar_points[:,0:3])
    # radar_colors = np.ones_like(radar_points[:,0:3])
    radar_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    lidar_pcd = o3d.geometry.PointCloud()
    lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points[:,0:3])
    lidar_colors = np.ones_like(lidar_points[:,0:3])
    lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_colors)

    
    if is_test_set:
        vod_labels = None
        o3d_labels = None 
    else:
        vod_labels = FrameLabels(frame_data.get_labels()).labels_dict
        o3d_labels = vod_to_o3d(vod_labels,vod_calib)    

    vod_preds = FrameLabels(frame_data.get_predictions()).labels_dict
    o3d_predictions = vod_to_o3d(vod_preds,vod_calib)
    

    vis_dict = {
        'radar_pcd': [radar_pcd],
        'lidar_pcd': [lidar_pcd],
        'vod_predictions': vod_preds,
        'o3d_predictions': o3d_predictions,
        'vod_labels': vod_labels,
        'o3d_labels': o3d_labels,
        'frame_id': frame_ids[frame_id]
    }
    return vis_dict



def set_camera_position(vis_dict,output_name):


    geometries = []
    geometries += vis_dict['radar_pcd']
    geometries += vis_dict['o3d_labels']

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280,height=720)    
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()

    o3d.io.write_pinhole_camera_parameters(f'{output_name}.json', param)
    vis.destroy_window()

def vis_one_frame(
    vis_dict,
    camera_pos_file,
    output_name,
    plot_radar_pcd=True,
    plot_lidar_pcd=False,
    plot_labels=False,
    plot_predictions=False):

    
    geometries = []
    name_str = ''

    if plot_radar_pcd:
        geometries += vis_dict['radar_pcd']
        point_size = 3
        name_str += 'Radar'
    if plot_lidar_pcd:
        geometries += vis_dict['lidar_pcd']
        point_size = 1 
        name_str += 'Lidar'
    if plot_labels:
        geometries += vis_dict['o3d_labels']
        name_str += 'GT'
    if plot_predictions:
        geometries += vis_dict['o3d_predictions']
        name_str += 'Pred'

    if name_str != '':
        output_name  = output_name / name_str
    output_name.mkdir(parents=True,exist_ok=True)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=640, height=480)
    # DRAW STUFF
    for geometry in geometries:
        viewer.add_geometry(geometry)
    
    # POINT SETTINGS
    opt = viewer.get_render_option()
    opt.point_size = point_size
    
    # BACKGROUND COLOR
    opt.background_color = np.asarray([0, 0, 0])

    # SET CAMERA POSITION
    ctr = viewer.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(camera_pos_file)    
    ctr.convert_from_pinhole_camera_parameters(parameters)
    
    # viewer.run()
    frame_id = vis_dict['frame_id']
    viewer.capture_screen_image(f'{output_name}/{frame_id}.png',True)
    viewer.destroy_window()



def vis_all_frames(
    kitti_locations,
    dt_path,
    CAMERA_POS_PATH,
    OUTPUT_IMG_PATH,
    plot_radar_pcd,
    plot_lidar_pcd,
    plot_labels,
    plot_predictions,
    is_test_set = False):

    
    if is_test_set:
        frame_ids  = [P(f).stem for f in glob(str(dt_path)+"/*")]

    else:
        pred_dict = get_pred_dict(dt_path)
        frame_ids = list(pred_dict.keys())

    for i in tqdm(range(len(frame_ids))):
        vis_dict = get_visualization_data(kitti_locations,dt_path,i,is_test_set)
        vis_one_frame(
            vis_dict = vis_dict,
            camera_pos_file=CAMERA_POS_PATH,
            output_name=OUTPUT_IMG_PATH,
            plot_radar_pcd=plot_radar_pcd,
            plot_lidar_pcd=plot_lidar_pcd,
            plot_labels=plot_labels,
            plot_predictions=plot_predictions)
        # break
    

# %%

def do_vis(tag,frames,is_test_set,test_dict,vod_data_path,resolution):
        abs_path = P(__file__).parent.resolve()
        base_path = abs_path.parents[1]
        CAMERA_POS_PATH = 'camera480v6.json'
        output_name = tag+'_testset' if is_test_set else tag 
        OUTPUT_IMG_PATH = base_path /'output' / 'vod_vis' / 'vis_video' /  (output_name + resolution+"v6")
    #--------------------------------------------------------------------------------

        OUTPUT_IMG_PATH.mkdir(parents=True,exist_ok=True)

        detection_result_path = base_path / path_dict[tag]

        dt_path = str(detection_result_path / 'result.pkl')    
        test_dt_path = base_path / test_dict[tag] if is_test_set else base_path / path_dict[tag]

        vis_path = test_dt_path if is_test_set else dt_path

        kitti_locations = get_kitti_locations(vod_data_path)
        
        
        # for tag in ['centerpoint_lidar','3dssd_lidar','pp_lidar','lidar_i']:
        #     do_vis(path_dict,tag)


        # UNCOMMENT THIS TO CREATE A CAMERA SETTING JSON,  
        # set_camera_position(vis_dict,'test_pos')


        # vis_dict = get_visualization_data(kitti_locations,dt_path,frame_id)
        # vis_one_frame(
        #     vis_dict = vis_dict,
        #     camera_pos_file=CAMERA_POS_PATH,
        #     output_name=OUTPUT_IMG_PATH,
        #     plot_radar_pcd=True,
        #     plot_lidar_pcd=True,
        #     plot_labels=True,
        #     plot_predictions=False)


        # IDS = ['00328','00312', '00339', '00338', '00336', '00335', '00334', '00333', '00340',
        # '00299', '00318', '00322', '00321', '00323', '00287', '00325', '00320', '00310',
        # '00408', '00305', '00302', '00297', '00296', '00294', '00293', '00290', '00199',
        # '00288', '00286', '00326',]        



        # raw_IDS2 = [8411, 4892, 4893, 4850,  487,  490, 8418, 8417, 4845,  491,  144, 8415,
        #     8414,  494,  495, 4851,  496, 8409, 8408, 4831, 4830, 4829, 4828, 4826,
        #     499, 4824, 4820, 4819, 4818, 4890]

        # IDS2 = [str(v).zfill(5) for v in raw_IDS2]
        # IDS = [ str(v).zfill(5) for v in list(range(110,171))]

        vis_subset(
            kitti_locations,
            vis_path,
            CAMERA_POS_PATH,
            OUTPUT_IMG_PATH,
            plot_radar_pcd=False,
            plot_lidar_pcd=True,
            plot_labels=False,
            plot_predictions=True,
            frame_ids=frames)
        

        # vis_subset(
        #     kitti_locations,
        #     vis_path,
        #     CAMERA_POS_PATH,
        #     OUTPUT_IMG_PATH,
        #     plot_radar_pcd=False,
        #     plot_lidar_pcd=True,
        #     plot_labels=False,
        #     plot_predictions=True,
        #     frame_ids=IDS2)
def vis_subset(
kitti_locations,
    dt_path,
    CAMERA_POS_PATH,
    OUTPUT_IMG_PATH,
    plot_radar_pcd,
    plot_lidar_pcd,
    plot_labels,
    plot_predictions,
    frame_ids):

    
    for i in tqdm(range(len(frame_ids))):
        vis_dict = get_visualization_data_true_frame(kitti_locations,dt_path,frame_ids[i],False)
        vis_one_frame(
            vis_dict = vis_dict,
            camera_pos_file=CAMERA_POS_PATH,
            output_name=OUTPUT_IMG_PATH,
            plot_radar_pcd=plot_radar_pcd,
            plot_lidar_pcd=plot_lidar_pcd,
            plot_labels=plot_labels,
            plot_predictions=plot_predictions)



# %%
def main():
    '''
    NOTE: EVERYTHING IS PLOTTED IN THE LIDAR FRAME 
    i.e. radar,lidar,gt,pred boxes all in lidar coordinate frame 
    '''

    vod_data_path = '/mnt/12T/public/view_of_delft'

    

    # ['centerpoint_lidar','3dssd_lidar','pp_lidar','lidar_i'] 

    test_dict = {
        'CFAR_lidar_rcsv':'/root/gabriel/code/parent/CenterPoint-KITTI/output/root/gabriel/code/parent/CenterPoint-KITTI/output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/IA-SSD-GAN-vod-aug-lidar/default/eval/epoch_5/val/default/final_result/data',
        'CFAR_radar':'output/root/gabriel/code/parent/CenterPoint-KITTI/output/IA-SSD-GAN-vod-aug/radar48001_512all/IA-SSD-GAN-vod-aug/default/eval/epoch_512/val/default/final_result/data',

    }
    


    
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    #------------------------------------SETTINGS------------------------------------
    frame_id = 333
    resolution_dict = {
        '720': [720, 1280]
    }
    resolution = '480'
    is_test_set = False
    tag = 'CFAR_lidar_rcs'
    
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    CAMERA_POS_PATH = 'camera480v6.json'
    output_name = tag+'_testset' if is_test_set else tag 
    OUTPUT_IMG_PATH = base_path /'output' / 'vod_vis' / 'vis_video' /  (output_name + resolution+"v6")
#--------------------------------------------------------------------------------

    OUTPUT_IMG_PATH.mkdir(parents=True,exist_ok=True)

    detection_result_path = base_path / path_dict[tag]

    dt_path = str(detection_result_path / 'result.pkl')    
    test_dt_path = base_path / test_dict[tag] if is_test_set else base_path / path_dict[tag]

    vis_path = test_dt_path if is_test_set else dt_path

    kitti_locations = get_kitti_locations(vod_data_path)

  


    vis_all_frames(
        kitti_locations,
        vis_path,
        CAMERA_POS_PATH,
        OUTPUT_IMG_PATH,
        plot_radar_pcd=False,
        plot_lidar_pcd=True,
        plot_labels=True,
        plot_predictions=False,
        is_test_set=is_test_set)


    # # #### CREATE DETECTION DATABASE #### 
    # tag_list = ['CFAR_lidar_rcs','lidar_i']
    # dt_paths = get_paths(base_path,path_dict,tag_list)
    # compare_models(kitti_locations,dt_paths,tag_list)

    # for tag in ['centerpoint_lidar','3dssd_lidar','pp_lidar','lidar_i']:
    #     tag_list = ['CFAR_lidar_rcs',tag]
    #     dt_paths = get_paths(base_path,path_dict,tag_list)
    #     compare_models(kitti_locations,dt_paths,tag_list)
    






    





    # # TODO: put this into a function 
    # test_path = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/CFAR_lidar_rcs1080_zoomedv2/LidarPred'
    # save_path = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/CFAR_lidar_rcs1080_zoomedv2_LidarPred.mp4'
    # dt_imgs = sorted(glob(str(P(test_path)/'*.png')))
    # make_vid(dt_imgs, save_path, fps=15)





#%%


def get_filtering_data(kitti_locations,dt_path,frame_id):
    pred_dict = get_pred_dict(dt_path)
    frame_ids = list(pred_dict.keys())
    frame_data = FrameDataLoader(kitti_locations,
                            frame_ids[frame_id],pred_dict)
    vod_calib = FrameTransformMatrix(frame_data)

    # print(len(frame_ids))
 
    vod_labels = FrameLabels(frame_data.get_labels()).labels_dict
   

    vod_preds = FrameLabels(frame_data.get_predictions()).labels_dict
    

    vis_dict = {
        'vod_predictions': vod_preds, 
        'vod_labels': vod_labels,
        'frame_id': frame_ids[frame_id]
    }
    return vis_dict





def gather_frames(frame_ids,det_path1,det_path2,gt_path,output_path):

  

    for frame in frame_ids:
        path = output_path / str(frame)
        os.makedirs(path,exist_ok=True)
        img1 = det_path1 + f"/{str(frame).zfill(5)}.png"
        img2 = det_path2 + f"/{str(frame).zfill(5)}.png"
        gt = gt_path + f"/{str(frame).zfill(5)}.png"
        
        os.symlink(img1,path / P(str(P(det_path1).parents[0].stem)+".png"))
        os.symlink(img2,path / P(str(P(det_path2).parents[0].stem)+".png"))
        os.symlink(gt,path / P("gt.png"))




def analyze_models(counter_path,frame_id_path,k):

    counter = np.load(counter_path)
    frame_ids = np.load(frame_id_path)

    model_counts = counter[:,:-1,:]
    gt_count = counter[:,-1,:]

    repeated_gt = np.repeat(np.expand_dims(gt_count,axis=1),model_counts.shape[1],axis=1)
    abs_diff = np.abs(repeated_gt-model_counts)
    difference_to_gt = np.sum(abs_diff,axis=2)



    relative_diff = np.abs(difference_to_gt[:,0]-difference_to_gt[:,1])
    diff_tens = torch.from_numpy(relative_diff)
    v,i = torch.topk(diff_tens,k)
    # ind = np.argpartition(relative_diff, -k)[-k:]

    return frame_ids[i.numpy()],v

def get_paths(base_path,path_dict,tag_list):
    dt_paths = []
    for tag in tag_list:
        detection_result_path = base_path / path_dict[tag]
        dt_path = str(detection_result_path / 'result.pkl')   
        dt_paths += [dt_path]
    return dt_paths

def compare_models(
    kitti_locations,
    dt_paths,
    tag_list,
    is_test_set = False):

    
    sample_dt = dt_paths[0]
    pred_dict = get_pred_dict(sample_dt)
    frame_ids = list(pred_dict.keys())
    
    # 1296 x num_models x 3:(car,ped,cyclist) 
    counter = np.zeros((len(frame_ids),len(dt_paths)+1,3))
    name_to_int = {'Car':0,'Pedestrian':1,'Cyclist':2}
    frame_ids_np = np.array([int(f) for f in frame_ids])

    for i in tqdm(range(len(frame_ids))):
        for j,dt_path in enumerate(dt_paths):
            vis_dict = get_visualization_data(kitti_locations,dt_path,i,is_test_set)
            detected_classes = [v['label_class'] for v in vis_dict['vod_predictions']]
            class_counter = Counter(detected_classes)
            for c in class_counter:
                counter[i][j][name_to_int[c]] = class_counter[c]
                # print("")

        gt_labels= [v['label_class'] for v in vis_dict['vod_labels'] if v['label_class'] in ['Car','Pedestrian','Cyclist']]
        gt_counter = Counter(gt_labels)
        for c in gt_counter:
                counter[i][-1][name_to_int[c]] = gt_counter[c]
    
    
    with open(f'{tag_list[0]}{tag_list[1]}.npy', 'wb') as f:
        np.save(f,counter)

    # with open('frame_ids_np.npy', 'wb') as f:
    #     np.save(f,frame_ids_np)




def generate_comparison_frames():
    
    
    frames = np.load('/root/gabriel/code/parent/CenterPoint-KITTI/tools/visual_utils/frames_to_generate.npy')
    
    frames = [str(f).zfill(5) for f in frames]

    tag = 'CFAR_lidar_rcs'

    is_test_set = False
    vod_data_path = '/mnt/12T/public/view_of_delft'
    resolution = '480'
    for t in [tag]+lidar_baseline_tags:
        do_vis(t,frames,is_test_set,None,vod_data_path,resolution)
    


#%%
if __name__ == "__main__":
    # RUN THESE COMMANDS FIRST
    # source py3env/bin/activate
    # export PYTHONPATH="${PYTHONPATH}:/root/gabriel/code/parent/CenterPoint-KITTI"
    generate_comparison_frames()
    # main()

# %%
