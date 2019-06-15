from vid_data import vid_data
import csv
import numpy as np
import os


def process_vid(vid):

    data_pts = []
    with open("generated_pts/full_data_opencv-{}.csv".format(vid['name'])) as f:
        rows = csv.DictReader(f)
        for row in rows:
            data_pt = {}
            for k, v in row.items():
                if ":" not in v and len(v.strip()) != 0:
                    v = np.float64(v)
                elif len(v.strip()) == 0:
                    v = None
                data_pt[k] = v
            data_pts.append(data_pt)

    if "accel_data" in vid:
        calib_data = []
        main_data = []
        with open(vid["accel_data"]) as f:
            lines = f.readlines()
            MODE = None
            for line in lines:
                if "calibration data" in line.lower():
                    MODE = 'CALIB'
                    continue
                if "base level" in line.lower() or "end data" in line.lower():
                    MODE = 'ignore'
                    continue
                if "=== begin data ===" in line.lower():
                    MODE = 'DATA'
                    continue

                if MODE == 'CALIB':
                    calib_data.append([np.int64(x.strip())
                                       for x in line.split('\t')])

                    # calib_data.append()
                elif MODE == 'DATA':
                    main_data.append([np.float64(x.strip())
                                      for x in line.split('\t')])
                    # print(line)

        avg_calib = np.average(calib_data, axis=0)
        ground = np.abs(np.max(avg_calib) - np.min(avg_calib))
        print("AVG", avg_calib)
        print("g accel:", ground)

        i = np.argmax(np.abs(avg_calib - np.median(avg_calib)))

        avg_calib[i] = np.median(avg_calib)
        print("New Avg", avg_calib)

        tot_avg = np.average(avg_calib)
        avg_calib -= tot_avg
        avg_calib /= ground
        print("Calibrated", avg_calib)

        new_data_pts = []
        main_data = main_data[::50]
        for data_pt in data_pts:
            if data_pt["time"] < vid["time_start"]:
                continue

            ac_t = data_pt["time"] - vid['time_start']

            closest_i = np.argmin([np.abs(x[0] - ac_t) for x in main_data])

            accel_data_close = main_data[closest_i][1:]
            accel_data_close -= tot_avg
            accel_data_close /= ground

            new_data = {**data_pt}
            new_data["acceleration"] = ":".join(
                [str(x) for x in accel_data_close])
            new_data_pts.append(new_data)
    else:
        new_data_pts = data_pts
    os.makedirs("generated_pts", exist_ok=True)

    with open("combined-data/all-{}.csv".format(vid["name"]), "w+", newline='') as f:
        headers = new_data_pts[0].keys()
        writer = csv.DictWriter(f, fieldnames=headers)

        writer.writeheader()
        for dp in new_data_pts:
            writer.writerow(dp)


if __name__ == "__main__":
    for i in range(len(vid_data)):
        process_vid(vid_data[i])
