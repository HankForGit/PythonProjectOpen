import argparse
import math
import os
import re
import signal
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Tuple, cast

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    ProgressCallback,
    Trainer,
    TrainingArguments,
)

# ================= 配置区域 =================
MODEL_ID = "/Users/hank/Desktop/Qwen3-8b"
DATA_PATH = "/Users/hank/PycharmProjects/PythonProject1/LLM1/try_qwen.json"
OUTPUT_DIR = "./unsloth_lora_output"

MAX_SEQ_LENGTH = 256
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4
EPOCHS = 20
FORCE_MPS = True
SEED = 42

EVAL_EVERY_STEPS = 10
EVAL_SAMPLE_SIZE = 3
SAVE_EVERY_STEPS = 10
SAVE_TOTAL_LIMIT = 10
STOP_SIGNAL_FILE = "./STOP_TRAINING_NOW"

# MPS 定向优化
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
            if isinstance(m, dict) and str(m.get("role") or "").strip() == "assistant"
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

    return prompt.strip(), answer.strip()


def clean_content(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 压缩多余空行，降低模型对连续换行符的过拟合。
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_qwen_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if isinstance(example.get("messages"), list):
        for item in example["messages"]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = clean_content(str(item.get("content") or ""))
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    if messages:
        return messages

    prompt, answer = extract_prompt_answer(example)
    prompt = clean_content(prompt)
    answer = clean_content(answer)
    if prompt and answer:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    return []


def encode_chat_example(tokenizer: Any, example: Dict[str, Any]) -> Dict[str, List[int]]:
    messages = build_qwen_messages(example)
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    last_assistant_idx = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            last_assistant_idx = i

    if last_assistant_idx <= 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prompt_messages = messages[:last_assistant_idx]
    answer_message = messages[last_assistant_idx]
    full_messages = prompt_messages + [answer_message]

    if not any(m["role"] == "user" for m in prompt_messages):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids: List[int] = tokenizer(
        prompt_text, add_special_tokens=False
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    )
    input_ids: List[int] = encoded["input_ids"]
    attention_mask: List[int] = encoded["attention_mask"]
    if not input_ids:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prompt_len = min(len(prompt_ids), len(input_ids))
    if prompt_len >= len(input_ids):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def find_last_checkpoint(output_dir: str) -> str | None:
    base = Path(output_dir)
    if not base.exists():
        return None
    checkpoints = []
    for path in base.glob("checkpoint-*"):
        suffix = path.name.split("checkpoint-")[-1]
        if suffix.isdigit():
            checkpoints.append((int(suffix), str(path)))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


class ManualStopController:
    def __init__(self, stop_signal_file: str) -> None:
        self.stop_signal_file = Path(stop_signal_file)
        self.stop_event = Event()
        self._signal_count = 0
        self._old_handlers: Dict[int, Any] = {}

    def _handle_signal(self, signum, frame) -> None:  # noqa: ANN001
        self._signal_count += 1
        if self._signal_count == 1:
            print("\n收到中断信号：将在当前 step 结束后保存 checkpoint 并停止训练。")
            self.stop_event.set()
            return
        raise KeyboardInterrupt

    def register(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except Exception:
                continue

    def restore(self) -> None:
        for sig, handler in self._old_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                continue
        self._old_handlers.clear()

    def should_stop(self) -> bool:
        if self.stop_event.is_set():
            return True
        if self.stop_signal_file.exists():
            print(f"\n检测到停止信号文件: {self.stop_signal_file}")
            try:
                self.stop_signal_file.unlink()
            except Exception:
                pass
            self.stop_event.set()
            return True
        return False


class ConsoleMetricsCallback(ProgressCallback):
    def __init__(self, stop_controller: ManualStopController) -> None:
        super().__init__()
        self.stop_controller = stop_controller
        self._stop_announced = False

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        # 关闭评估进度条，避免出现额外的 1/N 评估进度刷屏。
        return control

    def on_step_end(self, args, state, control, **kwargs):
        control = super().on_step_end(args, state, control, **kwargs)
        if self.stop_controller.should_stop():
            if not self._stop_announced:
                if state.is_world_process_zero and self.training_bar is not None:
                    self.training_bar.write("收到人工停止请求：正在保存 checkpoint 并结束训练...")
                self._stop_announced = True
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero or self.training_bar is None:
            return control
        if "loss" in logs and "eval_loss" not in logs:
            self.training_bar.write(
                str(
                    {
                        "loss": round(float(logs["loss"]), 6),
                        "grad_norm": round(float(logs.get("grad_norm", math.nan)), 6),
                        "learning_rate": float(logs.get("learning_rate", math.nan)),
                        "epoch": round(float(logs.get("epoch", state.epoch or 0.0)), 4),
                    }
                )
            )
            return control
        if "eval_loss" in logs:
            self.training_bar.write(str({"eval_loss": round(float(logs["eval_loss"]), 6)}))
            return control
        return control


def load_model_and_tokenizer(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    unsloth_ok = False
    model: Any | None = None
    if device == "cuda":
        try:
            from unsloth import FastLanguageModel  # type: ignore

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
                local_files_only=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                use_gradient_checkpointing="unsloth",
                random_state=SEED,
            )
            unsloth_ok = True
            print("已启用 Unsloth 加速路径。")
        except Exception as e:
            print(f"Unsloth 路径不可用，已回退标准 LoRA: {e}")

    if not unsloth_ok:
        model_dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=model_dtype,
            trust_remote_code=False,
            local_files_only=True,
        )
        if device != "cpu":
            model.to(device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
        if device == "mps":
            print("当前为 MPS，Unsloth 官方不支持，已使用 MPS 优化回退路径。")

    if model is None:
        raise RuntimeError("模型初始化失败：model 未成功构建。")
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="从最后 checkpoint 续训")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="从指定 checkpoint 路径续训，例如 ./unsloth_lora_output/checkpoint-50",
    )
    args_cli = parser.parse_args()

    device = pick_device(FORCE_MPS)
    print(f"使用设备: {device}")
    if device == "mps":
        torch.set_float32_matmul_precision("high")

    model, tokenizer = load_model_and_tokenizer(MODEL_ID, device)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

    raw_ds = cast(Dataset, load_dataset("json", data_files=DATA_PATH, split="train"))
    tokenized_all = raw_ds.map(
        lambda x: encode_chat_example(tokenizer, x),
        remove_columns=raw_ds.column_names,
    )
    tokenized_all = tokenized_all.filter(
        lambda x: len(x["input_ids"]) > 0 and any(t != -100 for t in x["labels"])
    )
    if len(tokenized_all) == 0:
        raise ValueError("样本量不足：训练集为空。")

    dropped = len(raw_ds) - len(tokenized_all)
    shuffled = tokenized_all.shuffle(seed=SEED)
    train_samples = shuffled
    sample_eval_size = min(EVAL_SAMPLE_SIZE, len(train_samples))
    eval_samples = train_samples.select(range(sample_eval_size))

    print(
        f"有效训练样本: {len(train_samples)} | 丢弃样本: {dropped} | 抽样评估样本: {len(eval_samples)}"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
    )
    use_pin_memory = device == "cuda"

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=EVAL_EVERY_STEPS,
        save_strategy="steps",
        save_steps=SAVE_EVERY_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
        group_by_length=False,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,
        dataloader_pin_memory=use_pin_memory,
        warmup_ratio=0.0167,
        weight_decay=0.0,
        max_grad_norm=1.0,
        disable_tqdm=False,
        fp16=False,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_samples,
        eval_dataset=eval_samples,
        data_collator=data_collator,
    )
    stop_controller = ManualStopController(STOP_SIGNAL_FILE)
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(ConsoleMetricsCallback(stop_controller))

    resume_ckpt: str | None = None
    resume_from_arg = getattr(args_cli, "resume_from", None)
    if resume_from_arg:
        candidate = os.path.abspath(resume_from_arg)
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"--resume_from 指定路径不存在: {candidate}")
        resume_ckpt = candidate
    elif args_cli.resume:
        resume_ckpt = find_last_checkpoint(OUTPUT_DIR)

    if resume_ckpt:
        print(f"检测到 checkpoint，继续训练: {resume_ckpt}")
    else:
        print("未检测到 checkpoint，从头训练。")

    stop_controller.register()
    print(
        f"人工停止方式: 按一次 Ctrl+C，或创建文件 {os.path.abspath(STOP_SIGNAL_FILE)}"
    )
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except KeyboardInterrupt:
        print("检测到强制中断，尝试立即保存当前训练状态...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        trainer.save_model(OUTPUT_DIR)
        state_saver = getattr(trainer, "save_state", None)
        if callable(state_saver):
            state_saver()
        else:
            trainer.state.save_to_json(os.path.join(OUTPUT_DIR, "trainer_state.json"))
    finally:
        stop_controller.restore()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"训练完成并保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
