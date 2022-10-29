label_color_palette_2d = {'Cyclist': (1, 0.0, 0.0),
                       'Pedestrian': (0.0, 0.4, 0.0),
                       'Car': (0.0, 0.0, 1.0),
                    #    'Car_pred': (0.4, 1.0, 1.0), # cyan/light blue
                       'Car_pred': (0.0, 0.0, 1.0), # cyan/light blue
                    #    'Pedestrian_pred': (0.8, 1.0, 0.6), # very light green
                       'Pedestrian_pred': (0.0, 1.0, 0.0), # very light green
                    #    'Cyclist_pred': (1.0, 0.7, 0.7),
                       'Cyclist_pred': (1.0, 0.0, 0.0),
                       'Others': (0.41,0.41,0.41),
                       'Truck': (0.41,0.41,0.41)}  # light red 

axis_length_3d = 1
axis_label_size = 0.5102, 255, 255

label_color_palette_3d = {
            "Car": 0xFF0000,
            "Pedestrian": 0x00FF00,
            "Cyclist": 0x0000FF,
            "bicycle": 0x0000FF,
            "DontCare": 0xAAAAAA,
            "moped_scooter": 0xAAAAAA,
            "rider": 0xFF0000,
            "bicycle_rack": 0xAAAAAA,
            "ride_other": 0xAAAAAA

}

label_line_width_3d = {
            "Car": 0.05,
            "Pedestrian": 0.05,
            "Cyclist": 0.02,
            "bicycle": 0.05,
            "DontCare": 0.01,
            "moped_scooter": 0.05,
            "rider": 0.02,
            "bicycle_rack": 0.01,
            "ride_other": 0.01
        }

radar_plot_color_3d = 0xFF0000
lidar_plot_color_3d = 0x0000FF

radar_velocity_color_3d = 0xFF0000

radar_pcl_size = 0.3
lidar_pcl_size = 0.02
