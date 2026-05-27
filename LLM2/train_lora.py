"""
LoRA 微调训练脚本。
基于 HuggingFace PEFT + Trainer，支持 MPS (Apple Silicon)。

用法:
    python train_lora.py
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    ProgressCallback,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments, BatchEncoding,
)

# 确保脚本能找到同目录下的 config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


# ================= 设备选择 =================

def pick_device(force_mps: bool = False) -> str:
    if force_mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("已设置 FORCE_MPS=True，但当前环境不可用 MPS。")
        return "mps"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ================= 数据处理 =================

def extract_prompt_answer(example: Dict[str, Any]) -> Tuple[str, str]:
    """从多种 JSON 格式中提取 prompt 和 answer。"""
    prompt = ""
    answer = ""

    # 格式 1: messages 列表
    if isinstance(example.get("messages"), list):
        user_msgs = [
            str(m.get("content") or "").strip()
            for m in example["messages"]
            if isinstance(m, dict) and str(m.get("role") or "").strip() == "user"
        ]
        assistant_msgs = [
            str(m.get("content") or "").strip()
            for m in example["messages"]
            if isinstance(m, dict) and str(m.get("role") or "").strip() == "assistant"
        ]
        if user_msgs and assistant_msgs:
            return user_msgs[-1], assistant_msgs[-1]

    # 格式 2: Alpaca (instruction/input/output)
    if "instruction" in example:
        prompt = str(example.get("instruction") or "")
        if example.get("input"):
            prompt = f"{prompt}\n{example['input']}"
        answer = str(example.get("output") or example.get("response") or "")
    elif "q" in example and "a" in example:
        prompt = str(example["q"])
        answer = str(example["a"])
    elif "question" in example and "answer" in example:
        prompt = str(example["question"])
        answer = str(example["answer"])
    elif "prompt" in example and "completion" in example:
        prompt = str(example["prompt"])
        answer = str(example["completion"])

    # 格式 3: 含"用户："和"AI："的纯文本
    if (not prompt or not answer) and isinstance(example.get("text"), str):
        text = example["text"]
        if "用户：" in text and "AI：" in text:
            try:
                user_part, ai_part = text.split("AI：", 1)
                prompt = user_part.replace("用户：", "").strip()
                answer = ai_part.strip()
            except ValueError:
                pass

    return prompt.strip(), answer.strip()


def build_record(tokenizer, example: Dict[str, Any]) -> Dict[str, str]:
    """将一条原始数据转为 {text, prompt, answer} 格式。"""
    # 已有 text 字段的直接用
    if isinstance(example.get("text"), str) and example["text"].strip():
        prompt, answer = extract_prompt_answer(example)
        return {"text": example["text"].strip(), "prompt": prompt, "answer": answer}

    # messages 格式
    if isinstance(example.get("messages"), list) and example["messages"]:
        cleaned_messages = []
        for item in example["messages"]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role in {"system", "user", "assistant"} and content:
                cleaned_messages.append({"role": role, "content": content})
        if cleaned_messages:
            text = tokenizer.apply_chat_template(
                cleaned_messages, tokenize=False, add_generation_prompt=False
            )
            prompt, answer = extract_prompt_answer({"messages": cleaned_messages})
            return {"text": text, "prompt": prompt, "answer": answer}

    # Alpaca 格式
    prompt, answer = extract_prompt_answer(example)
    if prompt and answer:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = f"用户：{prompt}\nAI：{answer}"
        return {"text": text, "prompt": prompt, "answer": answer}

    # 兜底
    parts = [str(v) for v in example.values() if isinstance(v, str) and v.strip()]
    return {"text": "\n".join(parts), "prompt": "", "answer": ""}


def load_datasets(tokenizer) -> Dataset:
    """从 config 中指定的多个 JSON 文件加载并合并数据。"""
    datasets_list = []
    for file_path in cfg.DATA_FILES:
        file_path = os.path.normpath(os.path.join(cfg.PROJECT_ROOT, file_path))
        if not os.path.exists(file_path):
            print(f"警告: 数据文件 {file_path} 不存在，跳过。")
            continue
        print(f"加载数据: {file_path}")
        raw = load_dataset("json", data_files=file_path, split="train")
        datasets_list.append(raw)

    if not datasets_list:
        raise FileNotFoundError("没有找到任何有效的数据文件，请检查 config.py 中的 DATA_FILES。")

    raw_ds = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    print(f"原始数据共 {len(raw_ds)} 条")

    processed = raw_ds.map(
        lambda x: build_record(tokenizer, x),
        remove_columns=raw_ds.column_names,
    )
    processed = processed.filter(
        lambda x: isinstance(x["text"], str) and x["text"].strip()
    )
    print(f"处理后有效数据: {len(processed)} 条")
    return processed


# ================= 自定义回调 =================

class DetailedProgressCallback(TrainerCallback):
    """每步训练后输出进度条，显示 step/epoch/loss/grad_norm/eval_ce/lr。"""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.pbar: Optional[tqdm] = None
        self._last_train_log: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        self.pbar = tqdm(
            total=self.total_steps, desc="训练进度", unit="step",
            dynamic_ncols=True, colour="green",
        )
        return control

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        return control

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> TrainerControl:
        if logs is None:
            return control

        if "loss" in logs and "eval_loss" not in logs:
            self._last_train_log = {
                "loss": float(logs.get("loss", float("nan"))),
                "grad_norm": float(logs.get("grad_norm", float("nan"))),
                "learning_rate": float(logs.get("learning_rate", 0.0)),
                "epoch": float(logs.get("epoch", state.epoch or 0.0)),
            }

        if "eval_loss" in logs:
            merged = {
                "step": float(state.global_step),
                "epoch": float(logs.get("epoch", self._last_train_log.get("epoch", 0.0))),
                "loss": self._last_train_log.get("loss", float("nan")),
                "grad_norm": self._last_train_log.get("grad_norm", float("nan")),
                "eval_cross_entropy": float(logs["eval_loss"]),
                "learning_rate": self._last_train_log.get("learning_rate", 0.0),
            }
            self.history.append(merged)
            self._update_pbar(merged)

        return control

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if state.global_step % args.eval_steps != 0 and self.pbar is not None:
            self.pbar.update(1)
        return control

    def _update_pbar(self, metrics: Dict[str, float]) -> None:
        if self.pbar is None:
            return
        self.pbar.update(1)
        self.pbar.set_postfix({
            "epoch": f"{metrics['epoch']:.2f}",
            "loss": f"{metrics['loss']:.4f}",
            "grad": f"{metrics['grad_norm']:.4f}",
            "eval_ce": f"{metrics['eval_cross_entropy']:.4f}",
            "lr": f"{metrics['learning_rate']:.2e}",
        })
        self.pbar.write(
            f"[step {int(metrics['step']):>5d} | epoch {metrics['epoch']:.2f}] "
            f"loss={metrics['loss']:.4f}  grad_norm={metrics['grad_norm']:.4f}  "
            f"eval_ce={metrics['eval_cross_entropy']:.4f}  lr={metrics['learning_rate']:.2e}"
        )


# ================= 主流程 =================

def main() -> None:
    # ---- 设备 ----
    device = pick_device(cfg.FORCE_MPS)
    print(f"使用设备: {device}")

    if cfg.MPS_FAST_MATH:
        os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ---- Tokenizer ----
    print("加载 tokenizer ...")
    tokenizer = cast(PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(cfg.MODEL_PATH, trust_remote_code=False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- 数据 ----
    dataset = load_datasets(tokenizer)
    if len(dataset) == 0:
        raise ValueError("训练集为空，无法训练。")

    # ---- 模型 ----
    print("加载模型 ...")
    model_dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_PATH,
        dtype=model_dtype,
        trust_remote_code=False,
    )
    if device != "cpu":
        model.to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ---- LoRA ----
    peft_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- Tokenize ----
    def tokenize_fn(examples: Dict[str, List[str]]) -> BatchEncoding:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.MAX_SEQ_LENGTH,
        )

    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- 训练参数 ----
    effective_batch = cfg.BATCH_SIZE * cfg.GRAD_ACCUM
    steps_per_epoch = (len(tokenized) + effective_batch - 1) // effective_batch
    total_steps = steps_per_epoch * cfg.EPOCHS
    print(f"每个 epoch 约 {steps_per_epoch} 步 | 总步数: {total_steps}")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_pin_memory = device == "cuda"

    args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        learning_rate=cfg.LEARNING_RATE,
        num_train_epochs=cfg.EPOCHS,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=cfg.EVAL_STEPS,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=use_pin_memory,
        warmup_ratio=cfg.WARMUP_RATIO,
        weight_decay=cfg.WEIGHT_DECAY,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        disable_tqdm=True,          # 由自定义回调接管
        fp16=False,
        bf16=use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        eval_dataset=tokenized,     # 小数据场景：同训练集评估
        data_collator=data_collator,
    )

    # 替换内置进度回调
    for cb_cls in (ProgressCallback, PrinterCallback):
        try:
            trainer.remove_callback(cb_cls)
        except Exception:
            pass
    detailed_progress = DetailedProgressCallback(total_steps=total_steps)
    trainer.add_callback(detailed_progress)

    # ---- 训练 ----
    print("\n开始训练 ...\n")
    trainer.train()

    # ---- 保存 ----
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(cfg.OUTPUT_DIR)
    tokenizer.save_pretrained(cfg.OUTPUT_DIR)
    print(f"\nLoRA 适配器已保存到: {cfg.OUTPUT_DIR}")

    # 评估报告
    last_eval = float("nan")
    step_logs = []
    for item in detailed_progress.history:
        step_logs.append({
            "step": int(item["step"]),
            "epoch": round(item["epoch"], 4),
            "loss": round(item["loss"], 6),
            "grad_norm": round(item["grad_norm"], 6),
            "eval_cross_entropy": round(item["eval_cross_entropy"], 6),
            "learning_rate": round(item["learning_rate"], 10),
        })
        last_eval = item["eval_cross_entropy"]

    report_path = os.path.join(cfg.OUTPUT_DIR, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"final_eval_loss": last_eval, "step_logs": step_logs}, f,
                  ensure_ascii=False, indent=2)
    print(f"评估报告已保存: {report_path}")
    print(f"最终 eval_loss: {last_eval:.6f}")


if __name__ == "__main__":
    main()
