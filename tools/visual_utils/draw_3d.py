# %%
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

# from vis_tools import fov_filtering





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
    radar_points = frame_data.radar_data
    radar_points = transform_pcl(radar_points,vod_calib.t_lidar_radar)
    radar_points = fov_filtering(radar_points,frame_ids[frame_id],is_radar=False)
    lidar_points = frame_data.lidar_data 
    lidar_points = fov_filtering(lidar_points,frame_ids[frame_id],is_radar=True)

    
    # convert into o3d pointcloud object
    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = o3d.utility.Vector3dVector(radar_points[:,0:3])
    # radar_colors = np.ones_like(radar_points[:,0:3])
    # radar_pcd.colors = o3d.utility.Vector3dVector(radar_colors)
    
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
        'o3d_predictions': o3d_predictions,
        'o3d_labels': o3d_labels,
        'frame_id': frame_ids[frame_id]
    }
    return vis_dict



def set_camera_position(vis_dict,output_name):


    geometries = []
    geometries += vis_dict['radar_pcd']
    geometries += vis_dict['o3d_labels']

    vis = o3d.visualization.Visualizer()
    vis.create_window()    
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
    plot_labels=True,
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
    viewer.create_window()
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

    

# %%

# %%
def main():
    '''
    NOTE: EVERYTHING IS PLOTTED IN THE LIDAR FRAME 
    i.e. radar,lidar,gt,pred boxes all in lidar coordinate frame 
    '''

    vod_data_path = '/mnt/12T/public/view_of_delft'

    path_dict = {
        'CFAR_radar':'output/IA-SSD-GAN-vod-aug/radar48001_512all/eval/best_epoch_checkpoint',
        'radar_rcsv':'output/IA-SSD-vod-radar/iassd_best_aug_new/eval/best_epoch_checkpoint',
        'radar_rcs':'output/IA-SSD-vod-radar/iassd_rcs/eval/best_epoch_checkpoint',
        'radar_v':'output/IA-SSD-vod-radar/iassd_vcomp_only/eval/best_epoch_checkpoint',
        'radar':'output/IA-SSD-vod-radar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'lidar_i':'output/IA-SSD-vod-lidar/all_cls/eval/checkpoint_epoch_80',
        'lidar':'output/IA-SSD-vod-lidar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcsv':'output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcs':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_rcs_only/eval/best_epoch_checkpoint',
        'CFAR_lidar_v':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_vcomp_only/eval/best_epoch_checkpoint',
        'CFAR_lidar':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_xyz_only/eval/best_epoch_checkpoint',
        'pp_radar_rcs' : 'output/pointpillar_vod_radar/debug_new/eval/checkpoint_epoch_80',
        'pp_radar_rcsv' : 'output/pointpillar_vod_radar/vrcomp/eval/best_epoch_checkpoint', 
        '3dssd_radar_rcs': 'output/3DSSD_vod_radar/rcs/eval/best_epoch_checkpoint',
        '3dssd_radar_rcsv': 'output/3DSSD_vod_radar/vcomp/eval/best_epoch_checkpoint',
        'centerpoint_radar_rcs': 'output/centerpoint_vod_radar/rcs/eval/best_epoch_checkpoint',
        'centerpoint_radar_rcsv': 'output/centerpoint_vod_radar/rcsv/eval/best_epoch_checkpoint',
        'second_radar_rcs': 'output/second_vod_radar/radar_second_with_aug/eval/checkpoint_epoch_80',
        'second_radar_rscv': 'output/second_vod_radar/pp_radar_rcs_doppler/eval/checkpoint_epoch_80',
        'pp_lidar': 'output/pointpillar_vod_lidar/debug_new/eval/checkpoint_epoch_80',
        '3dssd_lidar': 'output/3DSSD_vod_lidar/all_cls/eval/checkpoint_epoch_80',
        'centerpoint_lidar': 'output/centerpoint_vod_lidar/xyzi/eval/best_epoch_checkpoint'
    }


    test_dict = {
        'CFAR_lidar_rcsv':'/root/gabriel/code/parent/CenterPoint-KITTI/output/root/gabriel/code/parent/CenterPoint-KITTI/output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/IA-SSD-GAN-vod-aug-lidar/default/eval/epoch_5/val/default/final_result/data',
        'CFAR_radar':'output/root/gabriel/code/parent/CenterPoint-KITTI/output/IA-SSD-GAN-vod-aug/radar48001_512all/IA-SSD-GAN-vod-aug/default/eval/epoch_512/val/default/final_result/data',

    }
    
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    #------------------------------------SETTINGS------------------------------------
    frame_id = 333
    is_test_set = True
    tag = 'CFAR_lidar_rcsv'
    CAMERA_POS_PATH = 'test_pos.json'
    output_name = tag+'_testset' if is_test_set else tag 
    OUTPUT_IMG_PATH = base_path /'output' / 'vod_vis' / 'vis_video' /  output_name
    #--------------------------------------------------------------------------------

    OUTPUT_IMG_PATH.mkdir(parents=True,exist_ok=True)
    detection_result_path = base_path / path_dict[tag]

    dt_path = str(detection_result_path / 'result.pkl')    
    test_dt_path = base_path / test_dict[tag]

    kitti_locations = get_kitti_locations(vod_data_path)
    
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

    vis_all_frames(
        kitti_locations,
        test_dt_path,
        CAMERA_POS_PATH,
        OUTPUT_IMG_PATH,
        plot_radar_pcd=False,
        plot_lidar_pcd=True,
        plot_labels=False,
        plot_predictions=True,
        is_test_set=is_test_set)



    
    # TODO: put this into a function 
    # test_path = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/CFAR_lidar_rcsvtest/LidarPred'
    # save_path = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/CFAR_lidar_rcsvtest/CFAR_lidar_rcsvtest_.mp4'
    # dt_imgs = sorted(glob(str(P(test_path)/'*.png')))
    # make_vid(dt_imgs, save_path, fps=15)

#%%
if __name__ == "__main__":

    # source py3env/bin/activate
    #export PYTHONPATH="${PYTHONPATH}:/root/gabriel/code/parent/CenterPoint-KITTI"
    main()

# %%
