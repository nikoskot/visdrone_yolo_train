from ultralytics import YOLO


def main():
    model = YOLO("D:/Documents/experiments/mutli_object_detection_and_tracking/train_det_20261017/runs/det_exp_1/weights/best.pt")

    metrics = model.val(
        data="D:/Datasets/VisDrone2019_DET_YOLO/data.yaml",
        split="test",
        imgsz=960,
        batch=4,
        device=0,
        plots=True,
    )

    print(metrics)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"Per-class mAP50-95: {metrics.box.maps}")


if __name__ == "__main__":
    main()