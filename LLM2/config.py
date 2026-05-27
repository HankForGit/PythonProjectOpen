"""
训练 & 推理集中配置文件。
所有路径均为相对于项目根目录的路径，或绝对路径。
"""
import os

# ================= 项目根目录 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================= 模型 =================
MODEL_PATH = "/Users/hank/Desktop/Qwen3-8b"

# ================= LoRA =================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
# Qwen3/Llama 的线性层名称，"all-linear" 表示全部
LORA_TARGET_MODULES = "all-linear"

# ================= 数据（相对于项目根目录） =================
DATA_FILES = [
    "LLM1/train.json",
    "LLM1/wx对话.json",
]

# ================= 训练参数 =================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "LLM2", "main", "lora_output")
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 2          # 等效 batch_size = BATCH_SIZE * GRAD_ACCUM = 2
LEARNING_RATE = 2e-4
EPOCHS = 3
EVAL_STEPS = 10         # 每 N 步评估一次
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

# ================= 设备 =================
FORCE_MPS = True
MPS_FAST_MATH = True

# ================= 推理 =================
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
MPS_GREEDY_DECODING = True
