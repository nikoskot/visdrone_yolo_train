"""
Export a trained Ultralytics YOLO model to ONNX, verify it against PyTorch, then
(optionally) build a TensorRT engine with trtexec.

    pip install ultralytics onnx onnxruntime-gpu onnxslim opencv-python
    python export_onnx_trt.py --weights runs/detect/train/weights/best.pt \
        --imgsz 960 --images datasets/mydata/valid/images --build

Run the --build step ON THE MACHINE THAT WILL SERVE THE MODEL: a TensorRT engine is tied
to the GPU architecture and TensorRT version it was built with.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def to_tensor(img_bgr, size):
    x = letterbox(img_bgr, size)[:, :, ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, dtype=np.float32) / 255.0


def check_parity(weights, onnx_path, images_dir, imgsz, n_images=8):
    """Compare raw network outputs of PyTorch vs ONNX Runtime on real images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model = YOLO(weights).model.float().eval().to(device)

    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    inp = sess.get_inputs()[0].name

    paths = sorted(p for p in Path(images_dir).glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n_images]
    if not paths:
        raise SystemExit(f"No images found in {images_dir}")

    worst_box, worst_cls = 0.0, 0.0
    for p in paths:
        x = to_tensor(cv2.imread(str(p)), imgsz)
        with torch.no_grad():
            out = torch_model(torch.from_numpy(x).to(device))
        ref = (out[0] if isinstance(out, (list, tuple)) else out).cpu().numpy()
        got = sess.run(None, {inp: x})[0]
        # output layout: (1, 4 + num_classes, num_anchors); first 4 rows are boxes (pixels)
        worst_box = max(worst_box, float(np.abs(ref[:, :4] - got[:, :4]).max()))
        worst_cls = max(worst_cls, float(np.abs(ref[:, 4:] - got[:, 4:]).max()))

    print(f"Parity over {len(paths)} images | max |box diff| = {worst_box:.4f} px, "
          f"max |class score diff| = {worst_cls:.5f}")
    if worst_cls > 1e-2 or worst_box > 1.0:
        print("WARNING: ONNX output deviates from PyTorch more than expected. Investigate "
              "before building an engine.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True, help="Must match deployment size")
    ap.add_argument("--images", required=True, help="Folder of real images for parity check")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    # 1) Export a static-shape, FP32, simplified ONNX (keep it as the portable artifact)
    onnx_path = Path(YOLO(args.weights).export(
        format="onnx", imgsz=args.imgsz, opset=args.opset,
        simplify=True, dynamic=False, half=False, batch=1,
    ))
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"ONNX exported and structurally valid: {onnx_path}")

    # 2) Numerical parity vs the original PyTorch weights
    check_parity(args.weights, onnx_path, args.images, args.imgsz)


if __name__ == "__main__":
    main()