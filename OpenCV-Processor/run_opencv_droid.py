import cv2
import numpy as np
import imutils
from collections import deque
import os
import csv
from vid_data import vid_data
# from pyzbar import pyzbar


# RENDER_FPS = 60
CUR_VID = vid_data[-1]
print("Parsing {}".format(CUR_VID["name"]))

redLower1 = (142, 95, 105)
redUpper1 = (201, 217, 177)
greenLower1 = (56, 58, 103)
greenUpper1 = (91, 166, 140)
yellowLower1 = (0, 128, 120)
yellowUpper1 = (39, 255, 255)

# EQUIL_POSITION = [600, 700]


def nothing(x):
    return


FAST_MODE = False
CALIBRATION_EQ_MODE = False
RENDER = True


AC_CROSS_SQ = 0.297**2 + 0.21**2


# Currently keeps over the FPS

# redLower2 = (20, 0, 0)
# redUpper2 = (200, 255, 255)


def draw_arr(fr, x, y, r):
    cv2.line(fr, (x-r, y), (x+r, y), (255, 0, 0))
    cv2.line(fr, (x, y-r), (x, y+r), (255, 0, 0))


def add_color_track(name):
    cv2.createTrackbar('h'+name, 'frame2', 0, 255, nothing)
    cv2.createTrackbar('s'+name, 'frame2', 0, 255, nothing)
    cv2.createTrackbar('v'+name, 'frame2', 0, 255, nothing)


def get_color(name):
    return (cv2.getTrackbarPos("h"+name, "frame2"),
            cv2.getTrackbarPos("s"+name, "frame2"),
            cv2.getTrackbarPos("v"+name, "frame2"))


def get_track(name):
    return cv2.getTrackbarPos(name, "frame2")


def find_cross_dist_sq(rect_points):
    # print(rect_points)
    dists_from_top_left_squared = [
        pts[0]**2 + pts[1]**2 for pts in rect_points]

    top_left_i = np.argmin(dists_from_top_left_squared)
    bottom_right_i = np.argmax(dists_from_top_left_squared)

    top_left_x, top_left_y = rect_points[top_left_i]
    bottom_right_x, bottom_right_y = rect_points[bottom_right_i]

    return (top_left_y - bottom_right_y)**2 + (top_left_x - bottom_right_x)**2


PTS_AMOUNT = 200
data_pts = []

cap = cv2.VideoCapture(CUR_VID['file_path'])
fps = ("fps" in CUR_VID and CUR_VID["fps"]) or cap.get(cv2.CAP_PROP_FPS)
print("Detected FPS as:", fps)
has_red = "has_red" in CUR_VID and CUR_VID["has_red"]
hold_frame = False

last_few_ratios = deque(maxlen=int(fps))

if not FAST_MODE:
    cv2.namedWindow('frame')
    cv2.namedWindow('frame2')

ret, frame = None, None

# add_color_track("red_lower")
# add_color_track("red_upper")
# add_color_track("yellow_lower")
# add_color_track("yellow_upper")


crop_pos = CUR_VID['crop']
# equil_position = CUR_VID['equi_pos']
# equil_position[0] -= crop_pos[1][0]
# equil_position[1] -= crop_pos[0][0]


def with_colour_mask(frame, color_low, color_high):
    return cv2.dilate(cv2.erode(cv2.inRange(frame, color_low, color_high), None, iterations=1), None, iterations=1)


def find_angle(f1, f2):
    if f2 is None or f1 is None or len(f1) < 2 or len(f2) < 2:
        return None

    return np.arctan2((f1[0] - f2[0]), (f1[1] - f2[1]))


while cap.isOpened():

    if not hold_frame:
        ret, frame = cap.read()

    if frame is None:
        print("Cap finished")
        break

    g_adjusted_center = []
    rect_points = np.array([])
    r_adjusted_center = []

    if "flip" in CUR_VID:
        frame = cv2.flip(frame, CUR_VID["flip"])
    crop_frame = frame[crop_pos[0][0]:crop_pos[0][1],
                       crop_pos[1][0]:crop_pos[1][1]]
    # crop_frame = imutils.resize(crop_frame, 500)

    # print(frame.shape)
    # frame = frame[100:800, 800:1400]
    # frame = imutils.resize(frame, width=600)

    frame_i = cap.get(cv2.CAP_PROP_POS_FRAMES)
    time = frame_i / fps

    # if frame_i > 600:
    #     break

    # if frame_i % 8 != 0 and RENDER:
    # continue

    blurred = cv2.GaussianBlur(crop_frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    new_frame = crop_frame.copy()

    # construct a mask for the color "red", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    red_mask = with_colour_mask(
        hsv, redLower1, redUpper1)
    green_mask = with_colour_mask(hsv, greenLower1, greenUpper1)
    yellow_mask = with_colour_mask(hsv, yellowLower1, yellowUpper1)

    # mask2 = cv2.inRange(hsv, redLower2, redUpper2)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    center = None

    y_cnts = cv2.findContours(
        yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_cnts = imutils.grab_contours(y_cnts)
    if len(y_cnts) > 0:
        c_y = max(y_cnts, key=cv2.contourArea)
        rbox = cv2.minAreaRect(c_y)
        rect_points = cv2.boxPoints(rbox).astype(np.int32)
        cv2.drawContours(new_frame, [rect_points], -1,
                         (255, 0, 0), 1, cv2.LINE_AA)

        sq_cross = find_cross_dist_sq(rect_points)

        # m per pixel
        last_few_ratios.appendleft(np.sqrt(AC_CROSS_SQ/sq_cross))

        # m per pixel

        cur_ratio = np.median(last_few_ratios)

    # cv2.drawContours(new_frame, [c_y], -1, (0, 255, 0), 2)
    # cv2.rectangle(new_frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0))

    if has_red:
        r_cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

        r_cnts = imutils.grab_contours(r_cnts)
    g_cnts = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    g_cnts = imutils.grab_contours(g_cnts)
    # only proceed if at least one contour was found
    if len(g_cnts) > 0:
        # print("cnts")
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(g_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] != 0 and M["m00"] != 0:
            center_g = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            # update the points queue
            g_adjusted_center = center_g

        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        if has_red and len(r_cnts) > 0:
            c = max(r_cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            if M["m00"] != 0 and M["m00"] != 0 and center_g is not None:

                center_r = (int(M["m10"] / M["m00"]),
                            int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if ((center_r[0] - center_g[0])**2 + (center_r[1] - center_g[1])**2) * cur_ratio**2 < 0.2**2:

                    r_adjusted_center = center_r

    # cv2.imshow('frame', np.hstack((
    #     cv2.bitwise_and(new_frame, new_frame, mask=red_mask), new_frame)))

    def to_norm(x):
        return ":".join(map(str, x))

    if "red_top" in CUR_VID and CUR_VID["red_top"]:
        angle = find_angle(g_adjusted_center, r_adjusted_center)
    else:
        angle = find_angle(r_adjusted_center, g_adjusted_center)
    # Format <time> <ratio (m per pixel)> <green coords> <red coords> <rect coords>
    data_pts.append({"time": time, "m_per_pixel": cur_ratio, "g_center": to_norm(g_adjusted_center),
                     "r_center": to_norm(r_adjusted_center), "rect_points": to_norm(rect_points.flatten()), "angle": angle})

    if frame_i % fps == 0:
        if FAST_MODE:
            print("m per pixel", cur_ratio)
            print(frame_i / fps, "sec processed")
    # print(redLower1, redUpper1)

    if not FAST_MODE:
        if len(g_adjusted_center) == 2:
            new_frame = cv2.circle(new_frame, tuple(
                g_adjusted_center), 5, (0, 255, 0), thickness=-1)
        if len(r_adjusted_center) == 2:
            new_frame = cv2.circle(new_frame, tuple(
                r_adjusted_center), 5, (0, 0, 255), thickness=-1)
        # draw equilb point
        # draw_arr(new_frame, *equil_position, 10)
        cv2.imshow('frame', new_frame)
        print(angle)
        key_code = cv2.waitKey(5) & 0xFF

        if key_code == ord('h'):
            hold_frame = not hold_frame

        if key_code == ord('q'):
            break


os.makedirs("generated_pts", exist_ok=True)


with open("generated_pts/full_data_opencv-{}.csv".format(CUR_VID["name"]), "w+", newline='') as f:
    headers = data_pts[0].keys()
    writer = csv.DictWriter(f, fieldnames=headers)

    writer.writeheader()
    for dp in data_pts:
        writer.writerow(dp)


cap.release()
cv2.destroyAllWindows()
