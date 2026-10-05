"""解析 wandb 离线 run 文件，绘制 recon_loss 训练曲线。

run-*.wandb 使用 wandb 的 protobuf 定长记录格式，
通过 DataStore.open_for_scan + scan_record 读取。
"""
import glob
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_history(run_path):
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    hist = []
    run_files = glob.glob(os.path.join(run_path, "run-*.wandb"))
    for run_file in run_files:
        ds = DataStore()
        ds.open_for_scan(run_file)
        while True:
            try:
                rec = ds.scan_record()
            except Exception:
                rec = None
            if rec is None:
                break
            # scan_record 返回 (dtype, data)，data 是 protobuf 字节
            if isinstance(rec, tuple):
                _, raw = rec
                if not isinstance(raw, bytes):
                    continue
                data = pb.Record()
                try:
                    data.ParseFromString(raw)
                except Exception:
                    continue
            elif isinstance(rec, bytes):
                data = pb.Record()
                try:
                    data.ParseFromString(rec)
                except Exception:
                    continue
            else:
                data = rec
            if data.WhichOneof("record_type") == "history":
                for item in data.history.item:
                    if item.key in ("recon_loss", "summary"):
                        try:
                            val = float(json.loads(item.value_json))
                        except Exception:
                            val = float(item.value_json)
                        hist.append({"step": data.history.step.num,
                                     "recon_loss": val})
        ds.close()
    return hist


if __name__ == "__main__":
    run_path = sys.argv[1] if len(sys.argv) > 1 else (
        r"wandb\offline-run-20261005_145040-z0m1zi88")
    out_png = sys.argv[2] if len(sys.argv) > 2 else r"plots\recon_loss_curve.png"

    hist = extract_history(run_path)
    if not hist:
        print("[WARN] 未找到 history 记录")
        sys.exit(1)

    steps = [h["step"] for h in hist]
    recon = [h["recon_loss"] for h in hist]

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(steps, recon, linewidth=0.8, alpha=0.5, label="recon_loss / batch")
    y = np.array(recon)
    kern = np.ones(10) / 10
    y_avg = np.convolve(y, kern, mode="valid")
    x_avg = np.array(steps[: len(y_avg)])
    plt.plot(x_avg, y_avg, color="red", linewidth=2, label="recon_loss (moving avg)")
    plt.xlabel("global step")
    plt.ylabel("recon_loss (MSE)")
    plt.title("MLP reconstruction loss curve (toyUSW, 5000x20, 30 epochs)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] 解析到 {len(hist)} 个 history 记录")
    print(f"[OK] 曲线已保存: {out_png}")
    print(f"[OK] 首 step recon_loss = {recon[0]:.4f}, 末 step = {recon[-1]:.4f}")