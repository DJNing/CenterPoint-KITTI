# %%
import matplotlib.pyplot as plt
import pickle 
from pathlib import Path as P
from matplotlib.patches import Rectangle as Rec
import numpy as np
import sys
sys.path.append("/Users/gabrielchan/Desktop/code/CenterPoint-KITTI")
# from pcdet.utils import calibration_kitti
from vod.visualization.settings import label_color_palette_2d
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameLabels, FrameTransformMatrix
from vod.frame.transformations import transform_pcl
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from vod.visualization import Visualization3D
from skimage import io
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
        'Car': (0.0, 0, 1.0),
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
                             


def get_visualization_data(kitti_locations,dt_path,frame_id):

    pred_dict = get_pred_dict(dt_path)
    frame_ids = list(pred_dict.keys())
    frame_data = FrameDataLoader(kitti_locations,
                             frame_ids[frame_id],pred_dict)
    vod_calib = FrameTransformMatrix(frame_data)


    # get pcd
    radar_points = frame_data.radar_data
    radar_points = transform_pcl(radar_points,vod_calib.t_lidar_radar)
    lidar_points = frame_data.lidar_data 
    
    # convert into o3d pointcloud object
    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = o3d.utility.Vector3dVector(radar_points[:,0:3])
    radar_colors = np.ones_like(lidar_points[:,0:3])
    radar_pcd.colors = o3d.utility.Vector3dVector(radar_colors)
    
    lidar_pcd = o3d.geometry.PointCloud()
    lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points[:,0:3])
    lidar_colors = np.ones_like(lidar_points[:,0:3])
    lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_colors)

    # GET BOXES IN VOD FORMAT
    vod_preds = FrameLabels(frame_data.get_predictions()).labels_dict
    vod_labels = FrameLabels(frame_data.get_labels()).labels_dict

    # CONVERT TO O3D GEOMETRY OBJECT
    o3d_predictions = vod_to_o3d(vod_preds,vod_calib)
    o3d_labels = vod_to_o3d(vod_labels,vod_calib)

    vis_dict = {
        'radar_pcd': [radar_pcd],
        'lidar_pcd': [lidar_pcd],
        'o3d_predictions': o3d_predictions,
        'o3d_labels': o3d_labels
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
    plot_radar_pcd=False,
    plot_lidar_pcd=True,
    plot_labels=True,
    plot_predictions=False):

    
    geometries = []

    if plot_radar_pcd:
        geometries += vis_dict['radar_pcd']
        point_size = 3
    if plot_lidar_pcd:
        geometries += vis_dict['lidar_pcd']
        point_size = 1 
    if plot_labels:
        geometries += vis_dict['o3d_labels']
    if plot_predictions:
        geometries += vis_dict['o3d_predictions']

    
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
    viewer.capture_screen_image(f"{output_name}.png")
    viewer.destroy_window()

# %%
def main():
    '''
    NOTE: EVERYTHING IS PLOTTED IN THE LIDAR FRAME 
    i.e. radar,lidar,gt,pred boxes all in lidar coordinate frame 
    '''

    vod_data_path = '/Users/gabrielchan/Desktop/data/view_of_delft_PUBLIC'
    
    #------------------------------------SETTINGS------------------------------------
    frame_id = 333
    detection_result_path = P('/Users/gabrielchan/Desktop/data/pcdet')
    dt_path = str(detection_result_path / 'result.pkl')
    CAMERA_POS_PATH = 'camera_position.json'
    OUTPUT_IMG_PATH = 'output'
    #--------------------------------------------------------------------------------

    kitti_locations = get_kitti_locations(vod_data_path)
    vis_dict = get_visualization_data(kitti_locations,dt_path,frame_id)
    

    # UNCOMMENT THIS TO CREATE A CAMERA SETTING JSON,  
    # set_camera_position(vis_dict,'test_pos')

    vis_one_frame(
        vis_dict = vis_dict,
        camera_pos_file=CAMERA_POS_PATH,
        output_name=f'{OUTPUT_IMG_PATH}/{frame_id}_vis',
        plot_radar_pcd=False,
        plot_lidar_pcd=True,
        plot_labels=True,
        plot_predictions=False)


#%%
if __name__ == "__main__":
    main()

# %%
