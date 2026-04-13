import os
import torch

# ================= 自动检测设备 =================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ 检测到 CUDA 设备：{torch.cuda.get_device_name(0)}")
    USE_CUDA = True
    USE_MPS = False
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ 检测到 MPS 设备 (Apple Silicon)")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    USE_CUDA = False
    USE_MPS = True
else:
    device = torch.device("cpu")
    print("⚠ 未检测到 GPU，使用 CPU 训练（速度较慢）")
    USE_CUDA = False
    USE_MPS = False

print(f"使用设备：{device}")

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

# ================= 配置区域 =================
MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./lora_output_auto"

# ================= 1. 准备数据 =================
data = [
    {"text": "用户：你是谁？\nAI：我是你训练出来的 AI 助手。"},
    {"text": "用户：你会写代码吗？\nAI：当然，我可以帮你写 Python 代码。"},
    {"text": "用户：今天天气怎么样？\nAI：我无法联网，不知道具体天气，但希望你心情愉快。"},
] * 10

dataset = Dataset.from_list(data)
print(f"数据准备完毕，共 {len(dataset)} 条数据")

# ================= 2. 加载模型与 Tokenizer =================
print("正在加载模型...")

# 根据设备设置优化选项
if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn_impl = "sdpa"
elif USE_MPS:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    compute_dtype = torch.float16 if torch.backends.mps.is_bf16_supported() else torch.float32
    attn_impl = None  # MPS 不支持 flash attention
else:
    compute_dtype = torch.float32
    attn_impl = None

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 加载模型
model_kwargs = {
    "torch_dtype": compute_dtype,
    "trust_remote_code": True,
}
if attn_impl:
    model_kwargs["attn_implementation"] = attn_impl

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
model = model.to(device)

# 启用梯度检查点以节省内存
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ================= 3. 配置 LoRA =================
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ================= 4. 配置训练参数 =================
# 根据设备调整 batch size
if USE_CUDA:
    batch_size = 4
    optim_type = "adamw_torch"
elif USE_MPS:
    batch_size = 2
    optim_type = "adamw_torch"
else:
    batch_size = 1
    optim_type = "adamw_torch"

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    num_train_epochs=1,
    save_strategy="no",
    fp16=USE_CUDA and not torch.cuda.is_bf16_supported(),
    bf16=USE_CUDA and torch.cuda.is_bf16_supported(),
    optim=optim_type,
    gradient_checkpointing=True,
    group_by_length=True,
    dataloader_num_workers=0 if USE_MPS else 0,
    report_to="none"
)

# ================= 5. 开始训练 =================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
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
