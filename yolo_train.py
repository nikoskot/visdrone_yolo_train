from ultralytics import YOLO


def main():
    model = YOLO('yolov8s.pt')
    
    model.train(
        data='D:/Datasets/VisDrone2019_DET_YOLO/data.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,            # single GPU; multi-GPU DDP is flaky in notebooks
        project='train_detection',
        amp=False,
        save_period=5,       # extra snapshot every 5 epochs
        workers=2,
        cos_lr=True,
        close_mosaic=10,
        deterministic=False,  # faster, tiny reproducibility cost
        seed=0,
        # time=11,           # optional: stop gracefully after 11 h (overrides epochs)
    )
    
if __name__ == "__main__":
    main()