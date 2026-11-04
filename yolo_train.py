from ultralytics import YOLO


def main():
    model = YOLO('yolov8s.pt')
    
    model.train(
        data='D:/Datasets/VisDrone2019_DET_YOLO/data.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        project='train_detection',
        amp=False,
        save_period=5,
        workers=2,
        cos_lr=True,
        close_mosaic=10,
        deterministic=False,
        seed=0,
        # time=11,
    )
    
if __name__ == "__main__":
    main()