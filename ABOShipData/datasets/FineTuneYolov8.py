from ultralytics import YOLO
import optuna as op

if __name__ == '__main__':
    pass

def objective(trial):
    model_type = 'yolov8l.pt'
    model = YOLO(model_type)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    model.train(data='custom_data.yaml', batch=-1, workers=4, imgsz=960, epochs=10, name='yolov8l_custom', device=0, patience=10, pretrained=True, verbose=True, close_mosaic=0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0)
    trial.report(accuracy, epoch)
