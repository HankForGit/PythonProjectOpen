import os
import torch

# 检测 MPS 可用性
if not torch.backends.mps.is_available():
    raise RuntimeError("未检测到 MPS，请确认使用的是支持 Metal 的 Mac (M1/M2/M3 等) 且已安装支持 MPS 的 PyTorch 版本。")

# 设置 MPS 为默认设备
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ================= 配置区域 =================
# 使用 Qwen2.5-0.5B，因为它非常小，下载快，不容易爆显存，适合测试流程
MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./lora_output_mps"

# ================= 1. 准备数据 =================
# 这里为了演示，直接手动创建一个简单的数据集
# 实际使用时，请使用 load_dataset 加载你的 json/jsonl 文件
data = [
    {"text": "用户：你是谁？\nAI：我是你训练出来的 AI 助手。"},
    {"text": "用户：你会写代码吗？\nAI：当然，我可以帮你写 Python 代码。"},
    {"text": "用户：今天天气怎么样？\nAI：我无法联网，不知道具体天气，但希望你心情愉快。"},
    # ... 这里可以添加更多数据
] * 10  # 复制几份以便让训练跑起来不仅仅是一瞬间

dataset = Dataset.from_list(data)
print(f"数据准备完毕，共 {len(dataset)} 条数据")

# ================= 2. 加载模型与 Tokenizer =================
print("正在加载模型...")

# MPS 不支持某些 CUDA 特定的优化
# torch.backends.cuda.matmul.allow_tf32 = True  # MPS 不支持
# torch.backends.cudnn.allow_tf32 = True       # MPS 不支持
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

device = torch.device("mps")
print(f"使用设备：{device}")

# MPS 不支持 BitsAndBytes (4-bit 量化)，改用全精度或半精度加载
# 对于小模型如 0.5B，可以直接用 float16 加载
compute_dtype = torch.float16 if torch.backends.mps.is_bf16_supported() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# 处理填充符：很多新模型默认没有 pad_token，需要指定
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# MPS 不支持 device_map="auto"，需要手动指定设备
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    # attn_implementation="sdpa",  # MPS 不支持 flash_attention
)

# 将模型移动到 MPS 设备
model = model.to(device)

# 启用梯度检查点以节省内存
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ================= 3. 配置 LoRA =================
peft_config = LoraConfig(
    r=8,                  # LoRA 秩，通常 8, 16, 32, 64
    lora_alpha=16,        # alpha 通常是 r 的 2 倍
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # target_modules="all-linear" # 这是一个偷懒的好办法，对所有线性层进行微调，效果通常最好
    target_modules=["q_proj", "v_proj"] # 显式指定（针对 Attention 部分）
)

# 应用 LoRA 配置
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ================= 4. 配置训练参数 =================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, # 如果显存不够，改小这个
    gradient_accumulation_steps=4, # 累积梯度，模拟大 Batch Size
    learning_rate=2e-4,
    logging_steps=5,
    num_train_epochs=1,            # 演示只跑 1 个 epoch
    save_strategy="no",            # 演示不保存中间 checkpoint
    fp16=torch.backends.mps.is_bf16_supported(),  # MPS 上的混合精度
    bf16=False,                    # MPS 对 BF16 支持有限
    optim="adamw_torch",           # 使用原生 AdamW，paged_adamw_8bit 在 MPS 上可能不可用
    gradient_checkpointing=True,
    group_by_length=True,
    dataloader_num_workers=0,      # MPS 通常不使用多进程数据加载
    report_to="none"               # 不上传到 wandb
)

# ================= 5. 开始训练 (使用 SFTTrainer) =================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",     # 指定数据集中包含文本的列名
    max_seq_length=512,            # 截断长度
    tokenizer=tokenizer,
    args=args,
    packing=True,
)

print("开始训练...")
trainer.train()

# ================= 6. 保存模型 =================
print(f"训练完成，正在保存 LoRA 适配器到 {OUTPUT_DIR} ...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("保存成功！")
