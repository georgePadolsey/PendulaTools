import cv2
import numpy as np
import imutils
from collections import deques
from vid_data import vid_data
# from pyzbar import pyzbar


# RENDER_FPS = 60
CUR_VID = vid_data[3]

redLower1 = (72, 95, 105)
redUpper1 = (201, 217, 177)
greenLower1 = (56, 58, 103)
greenUpper1 = (91, 166, 140)
yellowLower1 = (0, 128, 120)
yellowUpper1 = (39, 255, 255)

# EQUIL_POSITION = [600, 700]


CUR_FPS = CUR_VID['fps']


def nothing(x):
    return


FAST_MODE = False
CALIBRATION_EQ_MODE = False
RENDER = True


AC_CROSS_SQ = 0.297**2 + 0.21**2


# Currently keeps over the FPS
last_few_ratios = deque(maxlen=int(CUR_FPS))
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


AMOUNT = 200
green_pts = []
red_pts = []

cap = cv2.VideoCapture(CUR_VID['file_path'])

hold_frame = False


if not FAST_MODE:
    cv2.namedWindow('frame')
    cv2.namedWindow('frame2')

ret, frame = None, None

# add_color_track("red_lower")
# add_color_track("red_upper")
# add_color_track("yellow_lower")
# add_color_track("yellow_upper")


created_equil_tracks = False

# frame_i = 0


crop_pos = CUR_VID['crop']
# equil_position = CUR_VID['equi_pos']
# equil_position[0] -= crop_pos[1][0]
# equil_position[1] -= crop_pos[0][0]

while cap.isOpened():

    if not hold_frame:
        ret, frame = cap.read()
    frame = imutils.resize(frame, 500)
    # print(frame.shape)
    # frame = frame[100:800, 800:1400]
    # frame = imutils.resize(frame, width=600)
    if frame is None:
        print("Frame finished")
        break

    frame_i = cap.get(cv2.CAP_PROP_POS_FRAMES)
    time = frame_i / CUR_FPS

    if frame_i % 8 != 0 and RENDER:
        continue

    crop_frame = frame[crop_pos[0][0]:crop_pos[0][1],
                       crop_pos[1][0]:crop_pos[1][1]]

    if not created_equil_tracks and CALIBRATION_EQ_MODE:
        cv2.createTrackbar("equil_x", "frame2", 0,
                           frame.shape[1], nothing)
        cv2.createTrackbar("equil_y", "frame2", 0,
                           frame.shape[0], nothing)

    if not CALIBRATION_EQ_MODE:
        blurred = cv2.GaussianBlur(crop_frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    new_frame = crop_frame.copy()

    # if CALIBRATION_EQ_MODE:
    #     equil_position = np.array([get_track("equil_x"), get_track("equil_y")])
    #     equil_position[0] -= crop_pos[1][0]
    #     equil_position[1] -= crop_pos[0][0]

    # redLower1 = get_color("red_lower")
    # redUpper1 = get_color("red_upper")

    # greenLower1 = (cv2.getTrackbarPos('1h', 'frame2'),
    #              cv2.getTrackbarPos('1s', 'frame2'),
    #              cv2.getTrackbarPos('1v', 'frame2'))

    # greenUpper1 = (cv2.getTrackbarPos('2h', 'frame2'),
    #              cv2.getTrackbarPos('2s', 'frame2'),
    #              cv2.getTrackbarPos('2v', 'frame2'))
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    red_mask = cv2.inRange(hsv, redLower1, redUpper1)
    if not CALIBRATION_EQ_MODE:
        green_mask = cv2.inRange(hsv, greenLower1, greenUpper1)
        yellow_mask = cv2.inRange(hsv, yellowLower1, yellowUpper1)

        # mask2 = cv2.inRange(hsv, redLower2, redUpper2)
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        center = None
        new_frame = crop_frame.copy()

        y_cnts = cv2.findContours(
            yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        y_cnts = imutils.grab_contours(y_cnts)
        if len(y_cnts) > 0:
            c_y = max(y_cnts, key=cv2.contourArea)
            rbox = cv2.minAreaRect(c_y)
            ptsx = cv2.boxPoints(rbox).astype(np.int32)
            cv2.drawContours(new_frame, [ptsx], -1,
                             (255, 0, 0), 1, cv2.LINE_AA)

            sq_cross = find_cross_dist_sq(ptsx)

            # m per pixel
            last_few_ratios.appendleft(np.sqrt(AC_CROSS_SQ/sq_cross))

        # m per pixel

        cur_ratio = np.median(last_few_ratios)

    # cv2.drawContours(new_frame, [c_y], -1, (0, 255, 0), 2)
    # cv2.rectangle(new_frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0))

    if not CALIBRATION_EQ_MODE:
        r_cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        r_cnts = imutils.grab_contours(r_cnts)
        g_cnts = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        g_cnts = imutils.grab_contours(g_cnts)
        # only proceed if at least one contour was found
        if len(g_cnts) > 0 and len(r_cnts) > 0:
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
                if radius > 10 and RENDER:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(new_frame, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(new_frame, center, 5, (0, 255, 0), -1)

                # update the points queue
                adjusted_center = np.float64(center_g)
                adjusted_center *= cur_ratio
                green_pts.append([time, *adjusted_center])
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(r_cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            if M["m00"] != 0 and M["m00"] != 0 and center_g is not None:
                center_r = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if ((center_r[0] - center_g[0])**2 + (center_r[1] - center_g[1])**2) * cur_ratio**2 < 0.2**2:
                   # draw the circle and centroid on the frame,
                   # then update the list of tracked points
                    if RENDER:
                        cv2.circle(new_frame, (int(x), int(y)), int(radius),
                                   (0, 255, 255), 2)
                        cv2.circle(new_frame, center, 5, (0, 0, 255), -1)

                    # update the points queue
                    adjusted_center = np.float64(center_r)
                    adjusted_center *= cur_ratio
                    red_pts.append([time, *adjusted_center])

    if len(green_pts) > 5 and len(red_pts) > 5:
        for i in range(1, min([len(green_pts), 300, len(red_pts)])):

            if green_pts[len(green_pts)-i-1] is None or green_pts[len(green_pts) - i - 2] is None:
                continue

            if red_pts[len(red_pts)-i-1] is None or red_pts[len(red_pts) - i - 2] is None:
                continue

            thickness = int(np.sqrt(AMOUNT / float(i + 1)) * 2.5)

            # print(tuple(green_pts[-i-1][1:]/cur_ratio))

            p1 = tuple(np.int64(green_pts[-i-1][1:]/cur_ratio))
            p2 = tuple(np.int64(green_pts[-i][1:]/cur_ratio))

            r1 = tuple(np.int64(red_pts[-i-1][1:]/cur_ratio))
            r2 = tuple(np.int64(red_pts[-i][1:]/cur_ratio))
            # equil_position[0] -= crop_pos[1][0]
            # equil_position[1] -= crop_pos[0][0]
            cv2.line(new_frame, p1, p2, (0, 255, 0), thickness)
            cv2.line(new_frame, r1, r2, (0, 0, 255), thickness)

    # cv2.imshow('frame', np.hstack((
    #     cv2.bitwise_and(frame, frame, mask=red_mask), frame)))

    if frame_i % 240 == 0:
        if FAST_MODE:
            print("m per pixel", cur_ratio)
            print(frame_i / 240, "sec processed")
    # print(redLower1, redUpper1)

    if not FAST_MODE:
        # draw equilb point
        # draw_arr(new_frame, *equil_position, 10)
        cv2.imshow('frame', new_frame)
        key_code = cv2.waitKey(5) & 0xFF

        if key_code == ord('f'):
            low_wait = True

        if key_code == ord('h'):
            hold_frame = not hold_frame

        if key_code == ord('q'):
            break


np.savetxt("pts-"+CUR_VID['name']+".txt", green_pts)

cap.release()
cv2.destroyAllWindows()
