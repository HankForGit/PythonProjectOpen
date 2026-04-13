import json
import os
from typing import Any, Dict, List, Tuple, cast

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    ProgressCallback,
    Trainer,
    TrainingArguments,
)

# ================= 配置区域 =================
MODEL_ID = "/Users/hank/Desktop/Qwen3-8b"
OUTPUT_DIR = "./lora_output"
DATA_PATH = "/Users/hank/Downloads/qwen.json"

MAX_SEQ_LENGTH = 256
PER_DEVICE_BATCH = 1

GRAD_ACCUM = 1
LEARNING_RATE = 1e-4
EPOCHS = 5
FORCE_MPS = True
MPS_FAST_MATH = True
MPS_MATMUL_PRECISION = "high"

SPLIT_SEED = 42
EVAL_EVERY_N_STEPS = 10

# MPS 上不支持 pin_memory，避免无效 warning
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
if MPS_FAST_MATH:
    # 用更激进的近似计算换速度，仅影响 MPS。
    os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")


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


def extract_prompt_answer(example: Dict[str, Any]) -> Tuple[str, str]:
    prompt = ""
    answer = ""

    if isinstance(example.get("messages"), list):
        user_msgs = [
            str(m.get("content") or "").strip()
            for m in example["messages"]
            if isinstance(m, dict) and str(m.get("role") or "").strip() == "user"
        ]
        assistant_msgs = [
            str(m.get("content") or "").strip()
            for m in example["messages"]
            if isinstance(m, dict)
            and str(m.get("role") or "").strip() == "assistant"
        ]
        if user_msgs and assistant_msgs:
            return user_msgs[-1], assistant_msgs[-1]

    if "instruction" in example:
        prompt = str(example.get("instruction") or "")
        if example.get("input"):
            prompt = f"{prompt}\n{example['input']}"
        answer = str(example.get("output") or example.get("response") or "")
    elif "q" in example and "a" in example:
        prompt = str(example.get("q") or "")
        answer = str(example.get("a") or "")
    elif "question" in example and "answer" in example:
        prompt = str(example.get("question") or "")
        answer = str(example.get("answer") or "")
    elif "prompt" in example and "completion" in example:
        prompt = str(example.get("prompt") or "")
        answer = str(example.get("completion") or "")

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


def build_record(tokenizer: Any, example: Dict[str, Any]) -> Dict[str, str]:
    if isinstance(example.get("text"), str) and example["text"].strip():
        prompt, answer = extract_prompt_answer(example)
        return {"text": example["text"].strip(), "prompt": prompt, "answer": answer}

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

    parts = [str(v) for v in example.values() if isinstance(v, str) and v.strip()]
    return {"text": "\n".join(parts), "prompt": "", "answer": ""}


def expand_multiturn_to_sft_rows(raw_ds: Dataset) -> Dataset:
    rows: List[Dict[str, Any]] = []
    for example in raw_ds:
        if not isinstance(example.get("messages"), list):
            rows.append(cast(Dict[str, Any], example))
            continue

        cleaned_messages: List[Dict[str, str]] = []
        for item in example["messages"]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role in {"system", "user", "assistant"} and content:
                cleaned_messages.append({"role": role, "content": content})

        if not cleaned_messages:
            continue

        built = 0
        for idx, msg in enumerate(cleaned_messages):
            if msg["role"] != "assistant":
                continue
            prefix = cleaned_messages[: idx + 1]
            if any(m["role"] == "user" for m in prefix[:-1]):
                rows.append({"messages": prefix})
                built += 1

        if built == 0:
            rows.append({"messages": cleaned_messages})

    if not rows:
        raise ValueError("数据集中未找到可用于训练的样本。")
    return Dataset.from_list(rows)


class MergedStepProgress(ProgressCallback):
    def __init__(self) -> None:
        super().__init__()
        self._pending_train_log: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        # 关闭评估进度条，避免每步训练后出现额外的 1/N。
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return control
        if self.training_bar is None:
            return control

        if "loss" in logs and "eval_loss" not in logs:
            train_log = {
                "loss": float(logs["loss"]),
                "grad_norm": float(logs.get("grad_norm", 0.0)),
                "learning_rate": float(logs.get("learning_rate", 0.0)),
                "epoch": float(logs.get("epoch", state.epoch or 0.0)),
            }
            self._pending_train_log = dict(train_log)
            eval_steps = int(args.eval_steps or 0) if getattr(args, "eval_steps", None) is not None else 0

            # 评估步由 eval 日志统一打印，避免同一步 loss/eval 重复两行。
            if eval_steps > 0 and state.global_step > 0 and state.global_step % eval_steps == 0:
                return control

            return super().on_log(args, state, control, logs=train_log, **kwargs)

        if "eval_loss" in logs:
            merged: Dict[str, float] = dict(self._pending_train_log)
            merged["eval_loss"] = float(logs["eval_loss"])
            merged["eval_cross_entropy"] = float(logs["eval_loss"])
            merged["epoch"] = float(logs.get("epoch", merged.get("epoch", state.epoch or 0.0)))
            merged["step"] = float(state.global_step)
            if "loss" not in merged:
                merged["loss"] = float("nan")

            self.history.append(
                {
                    "step": merged["step"],
                    "epoch": merged["epoch"],
                    "loss": merged["loss"],
                    "eval_cross_entropy": merged["eval_cross_entropy"],
                }
            )
            self._pending_train_log = {}
            return super().on_log(args, state, control, logs=merged, **kwargs)

        # 训练结束统计等非 step 日志保持默认输出。
        return super().on_log(args, state, control, logs=logs, **kwargs)


def main() -> None:
    device = pick_device(FORCE_MPS)
    print(f"使用设备: {device}")
    if device == "mps":
        torch.set_float32_matmul_precision(MPS_MATMUL_PRECISION)

    print("正在加载 tokenizer...")
    tokenizer: Any = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

    raw_ds = cast(Dataset, load_dataset("json", data_files=DATA_PATH, split="train"))
    expanded_ds = expand_multiturn_to_sft_rows(raw_ds)
    processed = expanded_ds.map(
        lambda x: build_record(tokenizer, x),
        remove_columns=expanded_ds.column_names,
    )
    processed = processed.filter(
        lambda x: isinstance(x["text"], str) and x["text"].strip()
    )

    if len(processed) == 0:
        raise ValueError("样本量不足：训练集为空。")

    train_samples = processed.shuffle(seed=SPLIT_SEED)

    print(f"原始对话: {len(raw_ds)} | 展开样本: {len(expanded_ds)} | 有效样本: {len(processed)}")
    print(f"训练集: {len(train_samples)} | 评估集: {len(train_samples)} (同训练集)")

    print("正在加载模型...")
    model_dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=model_dtype,
        trust_remote_code=False,
        local_files_only=True,
    )

    if device != "cpu":
        model.to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    tokenized_train = train_samples.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_samples.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    effective_micro_batch = PER_DEVICE_BATCH * GRAD_ACCUM
    updates_per_epoch = (
        len(tokenized_train) + effective_micro_batch - 1
    ) // effective_micro_batch
    print(f"每个 epoch 约优化步数: {updates_per_epoch} | 总优化步数: {updates_per_epoch * EPOCHS}")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_pin_memory = device == "cuda"

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=EVAL_EVERY_N_STEPS,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
        group_by_length=False,
        lr_scheduler_type= "cosine",
        dataloader_num_workers=0,
        dataloader_pin_memory=use_pin_memory,
        warmup_ratio=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        disable_tqdm=False,
        fp16=False,
        bf16=use_bf16,
        use_mps_device=(device == "mps"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_train,
        data_collator=data_collator,
    )
    merged_progress = MergedStepProgress()
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(merged_progress)

    print("开始训练...")
    trainer.train()

    last_eval_ce = float("nan")
    step_eval_ce: List[Dict[str, float]] = []
    for item in merged_progress.history:
        step_item = {
            "step": float(item["step"]),
            "epoch": float(item["epoch"]),
            "loss": float(item["loss"]),
            "eval_cross_entropy": float(item["eval_cross_entropy"]),
        }
        step_eval_ce.append(step_item)
        last_eval_ce = step_item["eval_cross_entropy"]

    print(f"最后一步评估交叉熵(eval_loss): {last_eval_ce:.6f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_cross_entropy": last_eval_ce,
                "step_eval_cross_entropy": step_eval_ce,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"验证报告已保存: {report_path}")

    print(f"训练完成，正在保存 LoRA 适配器到 {OUTPUT_DIR} ...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("保存成功！")


if __name__ == "__main__":
    main()
