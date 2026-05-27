"""
LoRA 模型推理脚本。
加载基座模型 + LoRA 适配器，合并后进行交互式对话。

用法:
    python infer_lora.py [--lora_path ./lora_output]
"""
import argparse
import os
import sys
import time
from typing import Dict, List, cast

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


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


def build_prompt(tokenizer, user_text: str) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": user_text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"用户：{user_text}\nAI："


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 模型推理")
    parser.add_argument(
        "--lora_path", type=str, default=cfg.OUTPUT_DIR,
        help="LoRA 适配器路径"
    )
    args = parser.parse_args()

    lora_path = os.path.normpath(os.path.join(os.path.dirname(__file__), args.lora_path))

    if not os.path.exists(lora_path):
        print(f"错误: LoRA 适配器路径不存在: {lora_path}")
        print("请先运行 train_lora.py 训练，或使用 --lora_path 指定路径。")
        sys.exit(1)

    device = pick_device(cfg.FORCE_MPS)
    print(f"使用设备: {device}")

    if cfg.MPS_FAST_MATH:
        os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ---- Tokenizer ----
    tokenizer = cast(PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(cfg.MODEL_PATH, trust_remote_code=False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- 基座模型 ----
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_PATH,
            dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_PATH,
            dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )

    if device != "cpu":
        base_model.to(device)

    # ---- 加载并合并 LoRA ----
    model = PeftModel.from_pretrained(base_model, lora_path)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.eval()

    print(f"模型加载完成，LoRA 已合并。")
    if device == "mps" and cfg.MPS_GREEDY_DECODING:
        print("MPS 快速模式：使用 greedy decoding。")

    # ---- 交互推理 ----
    print("输入内容开始对话，输入空行退出。\n")
    while True:
        try:
            user_text = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_text:
            break

        prompt = build_prompt(tokenizer, user_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        input_len = int(inputs["input_ids"].shape[-1])
        do_sample = not (device == "mps" and cfg.MPS_GREEDY_DECODING)
        if device == "mps":
            torch.mps.synchronize()

        gen_kwargs: Dict = dict(
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            do_sample=do_sample,
            num_beams=1,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = cfg.TEMPERATURE
            gen_kwargs["top_p"] = cfg.TOP_P

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)

        if device == "mps":
            torch.mps.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-6)
        gen_len = max(int(out[0].shape[-1]) - input_len, 0)
        print(f"[{gen_len / elapsed:.1f} tok/s]")

        gen_ids = out[0][input_len:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"AI: {answer}\n")


if __name__ == "__main__":
    main()
