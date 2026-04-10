import argparse
import csv
import math
import os

import cv2
import numpy as np
import pandas as pd
import lmfit
import matplotlib.pyplot as plt


def render_points(image, points, window_name):
    canvas = image.copy()
    for py, px in points:
        cv2.circle(canvas, (px, py), 4, (0, 0, 255), -1)
    cv2.imshow(window_name, canvas)

def click_and_store(event, x, y, flags, param):
    points = param["points"]
    image = param["image"]
    window_name = param["window_name"]

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([y, x])
        render_points(image, points, window_name)
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            points.pop()
            render_points(image, points, window_name)


def collect_points_for_image(image, window_name="image", mode_name=""):
    points = []
    preview = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    render_points(preview, points, window_name)
    cv2.setMouseCallback(
        window_name,
        click_and_store,
        {
            "points": points,
            "image": preview,
            "window_name": window_name,
        },
    )

    print(f"[mode] {mode_name}")
    print("[controls] Left click: add point | Right click: undo last point | Enter/n/Space: next image | q/Esc: quit")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, ord('n'), ord(' ')):
            return True, points
        if key in (27, ord('q')):
            return False, points

def split_array(a):
  result = []
  current = []
  for row in a:
    if (row == [0,0]).all():
      if len(current) > 0:
        result.append(current)
        current = []
    else:
      current.append(row)
  if len(current) > 0:
    result.append(current)
  return result

# define model functions
def f_dilate(x, lam_d):
    # x = [M, C, h]
    # M: markers
    # C: contact points
    # h: the height of contact points
    # calculate the distance between markers and contact points
    d = []
    for j in range((len(x[0])-2)//3):
        dx, dy = 0.0, 0.0
        i = 0
        while x[i,4] != 0.:
            g = np.exp(-(((x[:,0] - x[i,2+3*j]) ** 2 + (x[:,1] - x[i,3+3*j]) ** 2)) * lam_d)

            dx += x[i,4+3*j] * (x[:,0] - x[i,2+3*j]) * g
            dy += x[i,4+3*j] * (x[:,1] - x[i,3+3*j]) * g
            i+=1
        if j==0:
            d = np.hstack((dx,dy))
        else:
            d = np.hstack((d, np.hstack((dx,dy))))
    
    return d

def f_shear(x, lam_s):
    # x = [M, G, s]
    # M: markers
    # G: origin of contact area
    # s: displacement of object
    d = []
    for j in range((len(x[0])-2)//2):
        # calculate displacement
        g = np.exp(-(((x[:,0] - x[0,2+2*j]) ** 2 + (x[:,1] - x[0,3+2*j]) ** 2)) * lam_s)

        dx, dy = x[1,2+2*j] * g, x[1,3+2*j] * g
        if j==0:
            d = np.hstack((dx,dy))
        else:
            d = np.hstack((d, np.hstack((dx,dy))))

    return d

def f_twist(x, lam_t):
    # x = [M, G, theta]
    # M: markers
    # G: origin of contact area
    # theta: twist degree of object
    d = []
    for j in range((len(x[0])-2)//2):
        theta = x[1,2+2*j]

        g = np.exp(-(((x[:,0] - x[0,2+2*j]) ** 2 + (x[:,1] - x[0,3+2*j]) ** 2)) * lam_t)

        d_x = x[:,0] - x[0,2+2*j]
        d_y = x[:,1] - x[0,3+2*j]

        rotx = d_x * np.cos(theta) - d_y * np.sin(theta)
        roty = d_x * np.sin(theta) + d_y * np.cos(theta)  

        dx, dy =  (rotx - d_x) * g, (roty - d_y) * g

        if j==0:
            d = np.hstack((dx,dy))
        else:
            d = np.hstack((d, np.hstack((dx,dy))))

    return d

if __name__ == "__main__":

    calib_type = 0 # 0: dilate, 1: shear, 2: twist

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--folder", type=str, default="data/", help="folder containing images")
    args = argparser.parse_args()

    base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_folder = os.path.join(base_path, args.folder)
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

    ball_r = 3 / (0.0266*2)
    points_by_image = []

    M = []
    d_dx, d_dy = [], []
    s_dx, s_dy = [], []
    t_dx, t_dy = [], []
    while True:
        img_path = os.path.join(data_folder, f"{calib_type}.png")
        csv_path = os.path.join(data_folder, f"{calib_type}.csv")

        if not os.path.exists(csv_path):
            break
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image for calibration index {calib_type}: {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        mode = calib_type % 3
        if mode == 0:
            mode_name = "dilate image: click O (center), E (edge), then contact points C1..Cn"
        elif mode == 1:
            mode_name = "shear image: click O_1 only (shifted center)"
        else:
            mode_name = "twist image: click E_1 only (rotated edge point)"

        should_continue, image_points = collect_points_for_image(
            image,
            window_name="image",
            mode_name=mode_name,
        )
        if not should_continue:
            print("Calibration point collection stopped by user.")
            break

        if mode == 0 and len(image_points) < 2:
            raise ValueError("Dilation image requires at least 2 points: O and E.")
        if mode in (1, 2) and len(image_points) != 1:
            raise ValueError("Shear/Twist images require exactly 1 point each (O_1 or E_1).")

        points_by_image.append(image_points)

        print(f"points selected in image {calib_type}: {len(image_points)}")
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if calib_type % 3 == 0:
                    if calib_type == 0:
                        M.append([int(row[0]), int(row[1])])
                    d_dx.append(float(row[2]))
                    d_dy.append(float(row[3]))
                elif calib_type % 3 == 1:
                    s_dx.append(float(row[2]))
                    s_dy.append(float(row[3]))
                else:
                    t_dx.append(float(row[2]))
                    t_dy.append(float(row[3]))
        calib_type += 1

    cv2.destroyAllWindows()

    if calib_type == 0:
        raise RuntimeError(f"No calibration data found in folder: {data_folder}")

    M = np.asarray(M, dtype=float)
    if M.size == 0:
        raise RuntimeError("No marker coordinates were loaded from CSV files.")
    M = M.reshape(-1, 2)

    num_triplets = len(d_dx) // len(M)
    if len(d_dx) % len(M) != 0:
        raise RuntimeError("Invalid d_dx size: cannot infer number of calibration triplets.")
    if len(points_by_image) < num_triplets * 3:
        raise RuntimeError(
            f"Not enough point annotations. Need {num_triplets * 3} images (3 per triplet), got {len(points_by_image)}."
        )

    # obtian marker's real displacement under different loads
    d_d = np.array(np.hstack((d_dx,d_dy)))
    d_s = np.array(np.hstack((s_dx,s_dy))) - d_d
    d_t = np.array(np.hstack((t_dx,t_dy))) - d_d

    all_points = []
    for i in range(num_triplets):
        p_d = points_by_image[i * 3]
        p_s = points_by_image[i * 3 + 1]
        p_t = points_by_image[i * 3 + 2]

        # Group format expected by downstream code: [O, E, C..., O_1, E_1]
        grouped_points = [p_d[0], p_d[1], *p_d[2:], p_s[0], p_t[0]]
        all_points.append(np.array(grouped_points, dtype=float))

    if len(all_points) == 0:
        raise RuntimeError("No valid contact points were selected from image triplets.")

    Cd, Cd_p = [], []
    Cs, Cs_p = [], []
    Ct, Ct_p = [], []

    count = 0
    for p in all_points:
        if len(p) < 4:
            raise ValueError("Each point group must contain at least 4 points: O, E, O_1, E_1.")

        # the center pos of contact circle
        print(p)
        O = np.array(p[0], dtype=float)
        O_1 = np.array(p[-2], dtype=float)
        # calculate shear displacement
        s = O_1 - O
        # the contact edge point
        E = np.array(p[1], dtype=float)
        E_1 = np.array(p[-1], dtype=float)
        # calculate twist degree
        a2 = (E-O)[0]**2+(E-O)[1]**2
        b2 = (E_1-O)[0]**2+(E_1-O)[1]**2
        c2 = (E-E_1)[0]**2+(E-E_1)[1]**2
        denom = 2 * math.sqrt(a2) * math.sqrt(b2)
        if denom == 0:
            raise ValueError("Invalid geometry for theta computation: contact edge points overlap center.")
        cos_theta = max(-1.0, min(1.0, (a2 + b2 - c2) / denom))
        theta = math.acos(cos_theta)
        # contact points
        C = np.array(p[2:-2], dtype=float)
        # calculate height of contact points
        h = []
        for c in C:
            h_square = (E-O)[0]**2+(E-O)[1]**2-((c-O)[0]**2+(c-O)[1]**2)
            h.append([math.sqrt(max(h_square, 0.0))])

        if len(h) > len(M):
            raise ValueError("Number of selected contact points exceeds number of markers.")

        # concatenate contact points and corresponding height
        Cd = np.concatenate((C, np.array(h, dtype=float)),axis=1)
        Cd_zero = np.zeros((len(M)-len(h), 3))
        Cd = np.concatenate((Cd,Cd_zero),axis=0)

        # shear
        Cs = np.array([O, s], dtype=float)
        Cs_zero = np.zeros((len(M)-len(Cs), 2))
        Cs = np.concatenate((Cs,Cs_zero),axis=0)

        # twist
        Ct = np.array([O, [theta, 0.0]], dtype=float)
        Ct_zero = np.zeros((len(M)-len(Ct), 2))
        Ct = np.concatenate((Ct,Ct_zero),axis=0)

        if count==0:
            Cd_p = Cd.copy()
            Cs_p = Cs.copy()
            Ct_p = Ct.copy()
        else: 
            Cd_p = np.hstack((Cd_p, Cd))
            Cs_p = np.hstack((Cs_p, Cs))
            Ct_p = np.hstack((Ct_p, Ct))
        
        count += 1

    if count == 0:
        raise RuntimeError("No valid point groups were parsed from clicked points.")

    # fit data using lmfit to obtian optimal lambda
    f_d = np.concatenate((M, Cd_p),axis=1)
    model_d = lmfit.Model(f_dilate, independent_vars=['x'], param_names=['lam_d'])
    params_d = lmfit.Parameters()
    params_d.add('lam_d', value=0.1, min=0, max=1)
    results_d = model_d.fit(d_d, params=params_d, x=f_d)
    print("lam_d: ", results_d.params['lam_d'].value)
     
    f_s = np.concatenate((M, Cs_p),axis=1)
    model_s = lmfit.Model(f_shear, independent_vars=['x'], param_names=['lam_s'])
    params_s = lmfit.Parameters()
    params_s.add('lam_s', value=0.1, min=0, max=1)
    results_s = model_s.fit(d_s, params=params_s, x=f_s)
    print("lam_s: ", results_s.params['lam_s'].value)

    # twist concatenate
    f_t = np.concatenate((M, Ct_p),axis=1)
    model_t = lmfit.Model(f_twist, independent_vars=['x'], param_names=['lam_t'])
    params_t = lmfit.Parameters()
    params_t.add('lam_t', value=0.1, min=0, max=1)
    results_t = model_t.fit(d_t, params=params_t, x=f_t)
    print("lam_t: ", results_t.params['lam_t'].value)
