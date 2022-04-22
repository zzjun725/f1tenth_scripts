import cv2
import numpy as np
from scipy import ndimage
import csv
import yaml
import os


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


if __name__ == "__main__":
    # Get map name
    config_file = "./config/params.yaml"
    with open(config_file, 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
    input_map = parsed_yaml["map_name"]
    input_map_ext = parsed_yaml["map_img_ext"]

    # Read image
    img_path = "./maps/" + input_map + input_map_ext
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

    sim_map = output_img
    cv2.imwrite("./maps/" + input_map+'.png', ~sim_map)

    # Dilate & Erode
    kernel = np.ones((5, 5), np.uint8)
    output_img = cv2.dilate(output_img, kernel, iterations=1)
    
    # Show images
    if not show_result([input_img, output_img], title="input & output"):
        exit(0)

    # Separate outer bound and inner bound
    contours, hierarchy = cv2.findContours(output_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) in [3, 4]
    outer_bound = output_img.copy()
    cv2.fillPoly(outer_bound, pts=[contours[2]], color=(0, 0, 0))
    inner_bound = np.zeros_like(output_img)
    cv2.fillPoly(inner_bound, pts=[contours[2]], color=(255, 255, 255))
    if not show_result([outer_bound, inner_bound], title="outer & inner"):
        exit(0)

    # Euclidean distance transform
    outer_arr = 1 - (outer_bound / 255.0).astype(int)
    inner_arr = 1 - (inner_bound / 255.0).astype(int)
    outer_edt = ndimage.distance_transform_edt(outer_arr)
    inner_edt = ndimage.distance_transform_edt(inner_arr)
    path_y, path_x = np.where(np.abs(outer_edt - inner_edt) < 1)
    path = np.vstack((path_x, path_y)).T

    # Use finding contour to smooth path
    path_img = np.zeros_like(output_img)
    for idx in range(len(path)):
        cv2.circle(path_img, path[idx], 2, (255, 255, 255), 1)
    kernel = np.ones((3, 3), np.uint8)
    while True:
        # print(1)
        path_img = cv2.dilate(path_img, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(path_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 2 and hierarchy[0][-1][-1] == 0:
            break
    path_img = cv2.ximgproc.thinning(path_img)
    contours, hierarchy = cv2.findContours(path_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not show_result([path_img], title="track"):
        exit(0)
    smooth_path = np.squeeze(contours[0])

    # Plot final result
    res_img = cv2.cvtColor(~output_img, cv2.COLOR_GRAY2BGR)
    for idx in range(len(smooth_path) - 1):
        cv2.line(res_img, smooth_path[idx], smooth_path[idx + 1], (0, 0, 255), 1)
    cv2.line(res_img, smooth_path[-1], smooth_path[0], (0, 0, 255), 1)  # connect tail to head
    for pt in smooth_path:
        cv2.circle(res_img, pt, 2, (0, 0, 255), 1)
    if not show_result([res_img], title="track"):
        exit(0)

    # Calculate distance to left border (inner) and right border (outer)
    smooth_x, smooth_y = smooth_path[:, 0], smooth_path[:, 1]
    left_dist = inner_edt[smooth_y, smooth_x]
    right_dist = outer_edt[smooth_y, smooth_x]

    # Scale from pixel to meters, translate coordinates and flip y
    yaml_file = "./maps/" + input_map + ".yaml"
    with open(yaml_file, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(0)
    scale = parsed_yaml["resolution"]
    offset_x = parsed_yaml["origin"][0]
    offset_y = parsed_yaml["origin"][1]

    smooth_x = smooth_x * scale + offset_x
    smooth_y = (h - smooth_y) * scale + offset_y
    left_dist = left_dist * scale
    right_dist = right_dist * scale

    left_dist, right_dist = right_dist, left_dist  # flip left and right because y is flipped

    # Save result to csv file
    data = np.vstack((smooth_x, smooth_y, left_dist, right_dist)).T
    module = os.path.dirname(os.path.abspath(__file__))
    csv_path = './' + input_map + ".csv"
    with open(csv_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["# x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
        for line in data:
            csv_writer.writerow(line.tolist())
