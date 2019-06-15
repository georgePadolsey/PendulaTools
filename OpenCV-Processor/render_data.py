from vid_data import vid_data
import cv2
import imutils
import numpy as np
import csv
from collections import deque
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage


def read_csv(vid):
    data_pts = {}
    with open("combined-data/all-{}.csv".format(vid['name'])) as f:
        rows = csv.DictReader(f)
        for row in rows:
            data_pt = {}
            for k, v in row.items():
                if ":" in v:
                    v = np.float64(v.split(":"))
                elif len(v.strip()) != 0:
                    v = np.float64(v)
                else:
                    v = None
                data_pt[k] = v
            data_pts[data_pt["frame_i"]] = data_pt
    return data_pts


def render(vid):

    data_pts = read_csv(vid)
    angle_plots = []
    accel_plots = []
    cv2.namedWindow("main")
    print("Processing ", vid['name'])

    cap = cv2.VideoCapture(vid['file_path'])
    fps = ("fps" in vid and vid["fps"]) or cap.get(cv2.CAP_PROP_FPS)

    crop_pos = vid['crop']

    cap.set(cv2.CAP_PROP_POS_FRAMES, min(data_pts.keys()))

    TRAIL_LEN = 300
    green_buf = deque(maxlen=TRAIL_LEN)
    red_buf = deque(maxlen=TRAIL_LEN)

    first_run = True

    bottomPlotAddition = None
    sidePlotAddition = None

    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor='w')
    line, = ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_xlim([0, 10])
    ax.set_ylabel('Angle from Equilibrium (Radians)')
    ax.set_xlabel('Time (s)')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim([-np.pi, np.pi])  # setup wide enough range here
    plt.grid()

    fig2, ax2 = plt.subplots(figsize=(6.4, 4.8), facecolor='w')
    line2, = ax2.plot([1, 2, 3], [4, 5, 6])
    line3, = ax2.plot([1, 2, 3], [4, 5, 6])
    # line4, = ax2.plot([1, 2, 3], [4, 5, 6])
    leg = plt.legend([line2, line3], ['x-motion', 'y-motion'], loc=1)
    ax2.get_xaxis().tick_bottom()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.get_yaxis().tick_left()
    ax2.set_xlim([0, 10])
    ax2.set_ylabel('Acceleration (ms^-2)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim([-3, 3])

    # plt.box('off')
    # plt.tight_layout()
    # plt.ion()

    graphRGB = mplfig_to_npimage(fig)
    gh, gw, _ = graphRGB.shape

    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # print(frame.shape)
    # out = cv2.VideoWriter('output.avi', fourcc, 60,
    #                       (frame.shape[0], frame.shape[1]))

    while cap.isOpened():

        # if not hold_frame:
            #
        ret, frame = cap.read()

        if frame is None:
            print("Cap finished")
            break

        if "flip" in vid:
            frame = cv2.flip(frame, vid["flip"])
        frame = frame[crop_pos[0][0]:crop_pos[0][1],
                      crop_pos[1][0]:crop_pos[1][1]]

        frame_i = cap.get(cv2.CAP_PROP_POS_FRAMES)
        time = frame_i / fps

        data_pt = data_pts[frame_i]
        # print("common")
        # print(data_pt.keys())
        if time < vid["time_start"]:
            continue

        if time > 10 + vid["time_start"]:
            break

        aspect = 700/frame.shape[1]
        frame = imutils.resize(frame, 700)

        if first_run:
            bottomPlotAddition = np.zeros((300, frame.shape[1], 3), np.uint8)
            sidePlotAddition = np.zeros(
                (frame.shape[0]+bottomPlotAddition.shape[0], 400, 3), np.uint8)

        # print("Render")
        angle = None

        if data_pt['angle'] is not None:
            angle = data_pt["angle"]

            angle_plots.append(
                [data_pt["time"] - vid["time_start"], data_pt["angle"]])
        else:
            angle_plots.append([data_pt["time"] - vid["time_start"], None])
        if data_pt['r_center'] is not None:
            frame = cv2.circle(
                frame, tuple(np.int64(data_pt["r_center"] * aspect)), 5, (0, 0, 255), -1)
            red_buf.appendleft(tuple(np.int64(data_pt["r_center"] * aspect)))

        if data_pt['g_center'] is not None:
            frame = cv2.circle(
                frame, tuple(np.int64(data_pt["g_center"] * aspect)), 5, (0, 255, 0), -1)
            green_buf.appendleft(tuple(np.int64(data_pt["g_center"] * aspect)))

        if "acceleration" in data_pt and data_pt['acceleration'] is not None:
            accel_plots.append(
                [data_pt["time"] - vid["time_start"], data_pt["acceleration"]])
            # print(data_pt["acceleration"])

        if "rect_points" in data_pt and data_pt["rect_points"] is not None:
            # print(data_pt["rect_points"].reshape(-1, 2))
            rect_points = np.int64(
                data_pt["rect_points"].reshape(-1, 2) * aspect)
            frame = cv2.drawContours(
                frame, [rect_points], -1, (255, 0, 0), 1)

        g_length = min([len(green_buf), TRAIL_LEN])
        for i in range(g_length):

            if i + 1 >= len(green_buf):
                break

            frame = cv2.line(
                frame, green_buf[i], green_buf[i+1], (0, 255, 0), int((1-i/g_length) * 5) + 1)

        r_length = min([len(red_buf), TRAIL_LEN])
        for i in range(r_length):

            if i + 1 >= len(red_buf):
                break

            frame = cv2.line(
                frame, red_buf[i], red_buf[i+1], (0, 0, 255), int((1-i/r_length) * 5) + 1)

        # ax = plt.subplot(111)

        # print(accel_plots)

        line_x_data = []
        line2y_data = []
        line3y_data = []
        # for i, ac in enumerate(accel_plots):

        #     if angle_plots[i][1] is None:
        #         continue
        #     line_x_data.append(ac[0])
        #     ang = angle_plots[i][1]
        #     a_motion = ac[1][0]

        #     b_motion = np.sqrt(ac[1][1]**2 + ac[1][2]**2)

        #     x_motion = a_motion * np.sin(ang) + b_motion * np.cos(ang)
        #     y_motion = a_motion * np.cos(ang) + b_motion * np.sin(ang)

        #     # print(x_motion, y_motion)
        #     line2y_data.append(y_motion)
        #     line3y_data.append(x_motion)

        # print(line_x_data, line2y_data)
        # print(len(line2y_data), len(line_x_data))

        if frame_i % 2 == 0:
            pass
            # line.set_xdata([x[0] for x in angle_plots])
            # line.set_ydata([x[1] for x in angle_plots])
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            # line2.set_data(line_x_data, line2y_data)
            # line3.set_data(line_x_data[:], line3y_data)

            # fig2.canvas.draw()
            # fig2.canvas.flush_events()
            # plt.set_data(, [x[1] for x in angle_plots])

            # plt.draw()

        # orig_frame_shape = frame.shape[:]
        # frame = np.vstack([frame, bottomPlotAddition])

        # frame[orig_frame_shape[0]:bottomPlotAddition.shape[0]+orig_frame_shape[0],
        #       0:orig_frame_shape[1], :] = cv2.resize(mplfig_to_npimage(fig), (orig_frame_shape[1], bottomPlotAddition.shape[0]))

        # frame = np.hstack([frame, sidePlotAddition])

        # frame[0:orig_frame_shape[0],
        #       orig_frame_shape[1]:orig_frame_shape[1]+sidePlotAddition.shape[1], :] = cv2.resize(mplfig_to_npimage(fig2), (sidePlotAddition.shape[1], orig_frame_shape[0]))

        # if angle is not None:
        #     font = cv2.QT_FONT_NORMAL
        #     frame = cv2.putText(
        #         frame, 'Angle', (frame.shape[1] - 200, frame.shape[0]-30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        #     frame = cv2.ellipse(frame, (frame.shape[1] - 160, frame.shape[0]-180), (100, 100), 0, 90,
        #                         -np.degrees(data_pt["angle"])+90, (255, 255, 255), -1)

        #     frame = cv2.line(frame, (frame.shape[1] - 160, frame.shape[0] - 280),
        #                      (frame.shape[1] - 160, frame.shape[0] - 80), (255, 0, 0), 2)
# angle is here so can write to bottom right

        cv2.imshow("main", frame)
        # out.write(frame)

        cv2.waitKey(5)
    # out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    render(vid_data[7])
