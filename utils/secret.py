import os

# === Weights & Biases 配置 ===
# 默认使用匿名离线模式（不需要 wandb 账号，数据保存在本地 ./wandb 目录）。
# 如果想使用自己的 wandb 账号：
#   1. 取消下面三行注释，并填入真实值
#   2. 将 WANDB_MODE 设为 "online" 或删除这一行
#
# WANDB_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
# WANDB_PROJECT_NAME = "YOUR_PROJECT_NAME"
# WANDB_ENTITY = "YOUR_WANDB_ENTITY"

WANDB_API_KEY = None
WANDB_PROJECT_NAME = "M2N2-TSAD"
WANDB_ENTITY = None

# 未配置 Key 时使用离线模式，避免强制登录导致训练中断
if WANDB_API_KEY is None:
    os.environ.setdefault("WANDB_MODE", "offline")
