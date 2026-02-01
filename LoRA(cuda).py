import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

# ================= 配置区域 =================
# 使用 Qwen2.5-0.5B，因为它非常小，下载快，不容易爆显存，适合测试流程
MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./lora_output"

# ================= 1. 准备数据 =================
# 这里为了演示，直接手动创建一个简单的数据集
# 实际使用时，请使用 load_dataset 加载你的 json/jsonl 文件
data = [
    {"text": "用户：你是谁？\nAI：我是你训练出来的AI助手。"},
    {"text": "用户：你会写代码吗？\nAI：当然，我可以帮你写Python代码。"},
    {"text": "用户：今天天气怎么样？\nAI：我无法联网，不知道具体天气，但希望你心情愉快。"},
    # ... 这里可以添加更多数据
] * 10  # 复制几份以便让训练跑起来不仅仅是一瞬间

dataset = Dataset.from_list(data)
print(f"数据准备完毕，共 {len(dataset)} 条数据")

# ================= 2. 加载模型与Tokenizer =================
print("正在加载模型...")

# 配置 4-bit 量化加载 (QLoRA)，大幅降低显存占用
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# 处理填充符：很多新模型默认没有 pad_token，需要指定
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto", # 自动分配 GPU
    trust_remote_code=True
)

# ================= 3. 配置 LoRA =================
peft_config = LoraConfig(
    r=8,                  # LoRA 秩，通常 8, 16, 32, 64
    lora_alpha=16,        # alpha 通常是 r 的 2 倍
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules="all-linear" # 这是一个偷懒的好办法，对所有线性层进行微调，效果通常最好
    target_modules=["q_proj", "v_proj"] # 显式指定（针对 Attention 部分）
)

# ================= 4. 配置训练参数 =================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, # 如果显存不够，改小这个
    gradient_accumulation_steps=4, # 累积梯度，模拟大 Batch Size
    learning_rate=2e-4,
    logging_steps=5,
    num_train_epochs=1,            # 演示只跑 1 个 epoch
    save_strategy="no",            # 演示不保存中间 checkpoint
    fp16=True,                     # 开启混合精度训练
    optim="paged_adamw_8bit",      # 使用 8bit 优化器节省显存
    report_to="none"               # 不上传到 wandb
)

# ================= 5. 开始训练 (使用 SFTTrainer) =================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",     # 指定数据集中包含文本的列名
    max_seq_length=512,            # 截断长度
    tokenizer=tokenizer,
    args=args,
)

print("开始训练...")
trainer.train()

# ================= 6. 保存模型 =================
print(f"训练完成，正在保存 LoRA 适配器到 {OUTPUT_DIR} ...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("保存成功！")