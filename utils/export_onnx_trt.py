import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_path", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--workspace-mib", type=int, default=4096)
    args = ap.parse_args()
    
    # 3) TensorRT engine via trtexec (FP16 here; FP32 engine is the accuracy reference)
    args.onnx_path = Path(args.onnx_path)
    if args.fp16:
        engine_path = args.onnx_path.with_suffix(".fp16.engine")
    else:
        engine_path = args.onnx_path.with_suffix(".fp32.engine")
    cmd = [
        "trtexec",
        f"--onnx={args.onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace_mib}",
        f"--timingCacheFile={args.onnx_path.with_suffix('.timing.cache')}",
    ]
    if args.fp16: 
        cmd.append("--fp16")
    
    print("\nTensorRT build command:\n  " + " ".join(cmd))
    if shutil.which("trtexec") is None:
        raise SystemExit("trtexec not found on PATH (install the TensorRT package/container).")
    subprocess.run(cmd, check=True)  # prints latency/throughput stats at the end
    
if __name__ == '__main__':
    main()