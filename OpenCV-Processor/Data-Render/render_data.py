from ..vid_data.py import vid_data
import cv2
CUR_VID = vid_data[-2]
print("Processing ", CUR_VID['name'])


cap = cv2.VideoCapture(CUR_VID['file_path'])
fps = ("fps" in CUR_VID and CUR_VID["fps"]) or cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():

    if frame is None:
        print("Cap finished")
        break
