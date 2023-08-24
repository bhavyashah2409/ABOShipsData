from ultralytics import YOLO

if __name__ == '__main__':
    model_type = 'yolov8m.pt'
    model = YOLO(model_type)
    model.train(data='custom_data.yaml', batch=8, workers=4, imgsz=640, epochs=500, name='yolov8m_500epochs', device=0, patience=50, pretrained=True, verbose=True)
    # model.train(data='custom_data.yaml', batch=4, workers=4, imgsz=640, epochs=1, name='yolov8s_custom', device=0, patience=10, pretrained=True, verbose=True, close_mosaic=0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0)
