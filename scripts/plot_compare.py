"""解析多个 wandb 离线 run，将多个模型的 recon_loss 曲线画在同一张图上对比。"""
import glob
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_history(run_path):
    """返回按 step 排序的 {"step": int, "recon_loss": float} 列表。"""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    hist = []
    run_files = glob.glob(run_path + "/run-*.wandb")
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
    # 按 step 聚合并去重
    by_step = {}
    for h in hist:
        by_step[h["step"]] = h["recon_loss"]
    return [{"step": s, "recon_loss": by_step[s]} for s in sorted(by_step)]


def moving_avg(x, k=10):
    return np.convolve(x, np.ones(k) / k, mode="valid")


if __name__ == "__main__":
    runs = [
        ("MLP", r"wandb\offline-run-20261005_145040-z0m1zi88"),
        ("LSTMEncDec", r"wandb\offline-run-20261005_145136-3y0prxlr"),
    ]
    out_png = "plots/recon_loss_compare_MLP_LSTM.png"

    plt.figure(figsize=(12, 6))
    for label, run_path in runs:
        hist = extract_history(run_path)
        if not hist:
            print(f"[WARN] {label}: 无 history 记录，跳过")
            continue
        steps = [h["step"] for h in hist]
        recon = [h["recon_loss"] for h in hist]
        # 将 step 归一化到 epoch 维度（各自的 batch 数不同）
        # 用 step / 样本数，便于对比收敛速度
        plt.plot(steps, recon, linewidth=0.8, alpha=0.4,
                 label=f"{label} (batch)")
        y = np.array(recon)
        ys = moving_avg(y)
        xs = np.array(steps[: len(ys)])
        plt.plot(xs, ys, linewidth=2.2,
                 label=f"{label} (moving avg, final={recon[-1]:.4f})")
        print(f"[OK] {label}: {len(hist)} 记录, "
              f"首 = {recon[0]:.4f}, 末 = {recon[-1]:.4f}")

    plt.xlabel("global step")
    plt.ylabel("recon_loss (MSE)")
    plt.title("Model comparison: reconstruction loss (toyUSW, 5000x20, 30 epochs)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] 对比图已保存: {out_png}")