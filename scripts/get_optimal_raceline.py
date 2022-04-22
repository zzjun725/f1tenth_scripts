"""
read the centerline waypoint.csv under teh directory 'script', and output 'optimal_raceline.csv'    
"""
from distutils.command.build_scripts import first_line_re
import os
import csv
from turtle import ycor
from unicodedata import name
import fire
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import numpy.linalg as LA
import seaborn as sns
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
optimization_tool_dir = os.path.join(os.path.split(os.path.split(script_dir)[0])[0], 'global_racetrajectory_optimization')


def smooth_waypoints(filename='race_v1.csv', gap_thres = 0.3, interpolate_times = 3):
    if filename:
        output_name = filename.split('.')[0] + '_smooth' + '.csv'
        with open(os.path.join(script_dir, filename)) as f:
            waypoints = csv.reader(f)
            new_waypoints = []
            # import ipdb; ipdb.set_trace()
            for i, row in enumerate(waypoints):
                # row = row.split(',')                    
                if i == 1:
                    first_point = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                    x_old, y_old = float(row[0]), float(row[1])
                    r_old, l_old = float(row[2]), float(row[3])
                    new_waypoints.append(np.array([x_old, y_old, r_old, l_old]))
                    continue
                if i > 1:
                    x, y = float(row[0]), float(row[1])
                    r, l = float(row[2]), float(row[3])
                    if np.linalg.norm(np.array([x, y])-np.array([x_old, y_old])) > gap_thres:
                        interp_points = [np.array([x_old, y_old, r_old, l_old]), np.array([x, y, r, l])]
                        for _ in range(interpolate_times):
                            i = 0
                            n = len(interp_points)
                            while i < n:
                                interp_points.insert(i+1, (interp_points[i] + interp_points[i+1]) /2 )
                                i+=2
                        new_waypoints.extend(interp_points[1:])
                        # import ipdb; ipdb.set_trace()
                    else:
                        new_waypoints.append(np.array([x, y, r, l]))
                    x_old, y_old, r_old, l_old = x, y, r, l
            ## check the end point with the first
            print(f'origin wp number{i-1}')
            # import ipdb; ipdb.set_trace()
            if np.linalg.norm(np.array([x_old, y_old])-np.array(first_point)[:2]) > gap_thres:
                interp_points = [np.array([x_old, y_old, r_old, l_old]), np.array([first_point])]
                for _ in range(interpolate_times):
                    i = 0
                    n = len(interp_points)
                    while i < n:
                        interp_points.insert(i+1, (interp_points[i] + interp_points[i+1]).squeeze() /2 )
                        i+=2
                new_waypoints.extend(interp_points[1:-1])
            # import ipdb; ipdb.set_trace()
            print(f'new wp number{len(new_waypoints)}')
        with open(os.path.join(script_dir, output_name), 'w') as f:
            # f.write(f'# x_m,y_m,w_tr_right_m,w_tr_left_m\n')
            for waypoint in new_waypoints:
                x, y, r, l = waypoint
                f.write('%f, %f, %f, %f\n' % (x, y, r, l))
            f.close        

def centerline_to_optimal_raceline(filename='wp.csv', left=0.5, right=0.5, drift=True):
    input_wp_path = os.path.join(script_dir, filename)
    input_wp_path2 = os.path.join(optimization_tool_dir, 'inputs','tracks', 'wp_drift.csv')
    # print(os.path.exists(os.path.dirname(input_wp_path2)))
    if filename and drift:
        with open(os.path.join(script_dir, filename)) as f:
            centerline_log = csv.reader(f)
            with open(input_wp_path2, 'w') as ff:
                for i, row in enumerate(centerline_log):
                    ff.write('%f, %f, %f, %f\n' % (float(row[0]), float(row[1]), left, right))
    
    # print(input_wp_path)
    # print(input_wp_path2)
    
    os.system(f'cp {input_wp_path} {input_wp_path2}')
    os.chdir(optimization_tool_dir)
    # print(optimization_tool_dir)
    os.system('python main_globaltraj.py')
    # with open(os.path.join(optimization_tool_dir, 'main_globaltraj.py')) as f:
    #     # print(f.read())
    #     exec(f.read())
    #     time.sleep(1)
    #     pass
    # subprocess.run(['python', os.path.join(optimization_tool_dir, 'main_globaltraj.py')], shell=True)
    # time.sleep(30)
    output_path = os.path.join(optimization_tool_dir, 'outputs', 'traj_race_cl.csv')
    output_path2 = os.path.join(script_dir, 'optimal_raceline.csv')
    # print(output_path)
    # print(output_path2)
    os.system(f'cp {output_path} {output_path2}')

# centerline_to_optimal_raceline()


def draw_optimalwp(filename='wp.csv', ref_filename='wp_interp.csv'):
    x = []
    y = []

    with open(os.path.join(script_dir, filename)) as f:
        print(f'load {filename}')
        waypoints_file = csv.reader(f)
        for i, row in enumerate(waypoints_file):
            # import ipdb;
            # ipdb.set_trace()
            if i > 2:
                row = row[0].split(';')
                x.append(float(row[1]))
                y.append(float(row[2]))
    plt.plot(x, y, 'bo', markersize=0.5)
    
    ref_x = []
    ref_y = []

    with open(os.path.join(script_dir, ref_filename)) as f:
        print(f'load {ref_filename}')
        waypoints_file = csv.reader(f)
        for i, row in enumerate(waypoints_file):
            ref_x.append(float(row[0]))
            ref_y.append(float(row[1]))  

    plt.plot(ref_x, ref_y, '-ro', markersize=0.1)
    plt.show()



def PJcurvature(x,y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    refer to https://github.com/Pjer-zhang/PJCurvature for detail
    """
    t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
    t_b = LA.norm([x[2]-x[1],y[2]-y[1]])
    
    M = np.array([
        [1, -t_a, t_a**2],
        [1, 0,    0     ],
        [1,  t_b, t_b**2]
    ])

    a = np.matmul(LA.inv(M),x)
    b = np.matmul(LA.inv(M),y)

    kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    return kappa, [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)

def read_wp(wpfile='./wp/wp.csv', removelap=True, interp=False):
    wp_x = []
    wp_y = []
    last_x = 0
    with open(os.path.join(script_dir, wpfile)) as f:
        print(f'load {wpfile}')
        waypoints_file = csv.reader(f)
        for i, row in enumerate(waypoints_file):
            x_, y_ = float(row[0]), float(row[1])
            if abs(x_ - last_x) > 0.0001:
                wp_x.append(x_)
                wp_y.append(y_)
                last_x = x_

    if removelap:
        wp_x, wp_y = remove_overlap_wp(wp_x, wp_y)
    if interp:
        start_x, start_y = wp_x[0], wp_y[0]
        end_x, end_y = wp_x[-1], wp_y[-1]
        interp_x, interp_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        
        wp_x.insert(0, interp_x)
        wp_y.insert(0, interp_y)
        wp_x.append(interp_x)
        wp_y.append(interp_y)
    print(f'number of {wpfile} after remove overlap: {len(wp_x)}')
    return wp_x, wp_y

def remove_overlap_wp(x, y):
    start_p = np.array([x[0], y[0]])
    for i in range(len(x))[10:]:
        p = np.array([x[i], y[i]])
        if LA.norm(p-start_p) < 0.05:
            return x[:i], y[:i]
    return x, y

def draw_wp(wpfile='wp.csv', removelap=True):
    x, y = read_wp(wpfile, removelap)
    # plt.plot(x, y, 'o', markersize=0.5)
    sns.scatterplot(x=x, y=y, hue=len(x) - np.arange(len(x)))
    plt.show()

def visualize_curvature_for_wp(wpfile='./wp/wp.csv'):
    wp_x, wp_y = read_wp(wpfile, removelap=True)
    kappa = []
    no = []
    po = []
    ka = []
    for idx in range(len(wp_y))[1:-2]:
        x = wp_x[idx-1:idx+2]
        y = wp_y[idx-1:idx+2]
        kappa, norm = PJcurvature(x,y)
        ka.append(kappa)
        no.append(norm)
        po.append([x[1],y[1]])

    po = np.array(po)
    no = np.array(no)
    ka = np.array(ka)
        
    fig=plt.figure(figsize=(8,5),dpi=120)
    ax=fig.add_subplot(2, 1, 1)
    plt.plot(po[:,0],po[:,1])
    plt.quiver(po[:,0],po[:,1],ka*no[:,0],ka*no[:,1])
    plt.axis('equal')

    ax=fig.add_subplot(2, 1, 2)
    plt.plot(ka, '-bo', markersize=0.1)
    plt.show()

def velocity_interp(wpfile='./wp/wp.csv', max_a = 8.0, max_v = 8.0, min_v = 2.0, half_windowsize=3, wp_gap=2):
    wp_name = os.path.basename(wpfile)
    wp_x, wp_y = read_wp(wpfile, removelap=True)
    output_path = os.path.join(script_dir, 'wp', 'interp_' + os.path.basename(wpfile))
    output_f = open(output_path, 'w')
    kappa = []
    wp_v = []

    # wp_vv = np.zeros((len(wp_x)-3, len(wp_x)-3))
    for idx in range(len(wp_y))[1:-1]:
        x = wp_x[idx-1:idx+2]
        y = wp_y[idx-1:idx+2]
        ka, norm = PJcurvature(x,y)
        kappa.append(ka)
        v = np.clip(np.sqrt(max_a/np.clip(abs(ka), 0.1, 3)), min_v, max_v)
        wp_v.append(v)
        # wp_vv[idx-1][idx-1] = v

    wp_v.insert(0, wp_v[0]); kappa.insert(0, kappa[0])
    wp_v.append(wp_v[-1]); kappa.append(kappa[-1])
    # wp_v.append(wp_v[-1]); kappa.append(kappa[-1])
    # wp_x = wp_x[1:-1]
    # wp_y = wp_y[1:-1]
    
    wp_x = wp_x[::wp_gap]
    wp_y = wp_y[::wp_gap]
    wp_v = wp_v[::wp_gap]
    kappa = kappa[::wp_gap]
    wp_v_num = len(wp_v)

    print(f'final {wp_name} number: {wp_v_num}')
    wp_v_smooth = []
    for idx in range(len(wp_x)):
        v_smooth = np.mean(wp_v[max(0, idx-half_windowsize): min(wp_v_num, idx+half_windowsize)])
        output_f.write('%f, %f, %f\n' % (wp_x[idx], wp_y[idx], v_smooth))
        wp_v_smooth.append(v_smooth)
    output_f.close()

    os.system(f'cp {output_path} ../src/wp_log/{os.path.basename(output_path)}')
    print('move interp_wp to sim_ws')
    # draw
    fig=plt.figure()
    ax=fig.add_subplot(2, 1, 1, projection='3d')
    # xx, yy=np.meshgrid(wp_x[1:-2], wp_y[1:-2])
    # ax.bar3d(xx.ravel(), yy.ravel(), 0, 0.01, 0.01, wp_vv.ravel(), shade=True)
    ax.scatter(wp_x, wp_y, wp_v_smooth, cmap='jet')
    
    ax=fig.add_subplot(2, 1, 2)
    plt.plot(wp_v, '--r', linewidth=0.5)
    plt.plot(wp_v_smooth, '-bo', markersize=0.1)
    plt.plot(kappa, '-g', linewidth=0.5)
    plt.show()




def show_result(imgs, title):
    if not imgs:
        return False
    height, width = imgs[0].shape[:2]
    w_show = 800
    scale_percent = float(w_show / width)
    h_show = int(scale_percent * height)
    dim = (w_show, h_show)
    img_resizes = []
    for img in imgs:
        img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_resizes.append(img_resize)
    img_show = cv2.hconcat(img_resizes)
    cv2.imshow(title, img_show)

    print("Press Q to abort / other keys to proceed")
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        return False
    else:
        cv2.destroyAllWindows()
        return True

def fix_slam_map_for_sim(mapname='race_v1'):
    # Read image
    img_path = os.path.join('./maps', mapname+'.pgm')
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    input_img = input_img[:, 80:]
    print(input_img.shape)
    h, w = input_img.shape[:2]

    # Flip black and white
    output_img = ~input_img

    # Convert to binary image
    ret, output_img = cv2.threshold(output_img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    # Find contours and only keep larger ones
    contours, hierarchy = cv2.findContours(output_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 70:
            cv2.fillPoly(output_img, pts=[contour], color=(0, 0, 0))

    show_result([input_img, output_img], title="input & output")
    sim_map = output_img
    output_path = os.path.join('./maps', mapname+'.png')
    cv2.imwrite(output_path, ~sim_map)    
    
    os.system(f'cp {output_path} ../src/f1tenth_gym_ros/maps/{mapname}.png')
    map_yaml_path = os.path.join('./maps', mapname+'.yaml')
    os.system(f'cp {map_yaml_path} ../src/f1tenth_gym_ros/maps/{mapname}.yaml')


if __name__ == '__main__':
    fire.Fire({
        'optim_raceline': centerline_to_optimal_raceline,
        'draw_wp': draw_wp,
        'draw_optimal': draw_optimalwp,
        'smooth_wp': smooth_waypoints,
        'vis_curv': visualize_curvature_for_wp,
        'interp_v': velocity_interp,
        'slammap_fix': fix_slam_map_for_sim
    })
