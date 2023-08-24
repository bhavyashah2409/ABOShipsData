import cv2 as cv
from ultralytics import YOLO

video = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_6.mp4"
# frames = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\frames\video_41"

model = YOLO(r'runs\detect\yolov8m_500epochs\weights\best.pt')

results = model(video, stream=True)
for result in results:
    classes = result.names
    frame = result.orig_img
    bboxes = result.boxes.data.detach().cpu().numpy().tolist()
    for xmin, ymin, xmax, ymax, p, c in bboxes:
        c = classes[int(c)]
        p = round(p * 100, 2)
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        frame = cv.putText(frame, f'{c}:{p}', (xmin, ymin - 10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    cv.imshow('Frame', frame)
    cv.waitKey(1)
