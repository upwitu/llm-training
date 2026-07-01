from tqdm import tqdm
import datetime
import json
import shutil
import os
import yaml
import re
import hashlib

# Disable Triton autotuning for MoE
os.environ['UNSLOTH_MOE_DISABLE_AUTOTUNE'] = '1'

CONFIG_PATH = "ddp_config.yml"

def resolve_env_vars(value: str) -> str:
    if isinstance(value, str):
        return re.sub(r'\$\{([^}]+)\}', lambda m: os.getenv(m.group(1), ""), value)
    return value

def get_cache_dir_from_config(config_path: str) -> str:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("cache:"):
                    next_line = next(f, "").strip()
                    if next_line.startswith("dir:"):
                        dir_val = next_line.split(":", 1)[1].strip().strip("\"'")
                        return resolve_env_vars(dir_val)
    except Exception as e:
        print(f"⚠️  Failed to read cache.dir from {config_path}: {e}")
    return os.getenv("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))

HF_CACHE_DIR = get_cache_dir_from_config(CONFIG_PATH)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["DATASETS_CACHE"] = HF_CACHE_DIR

# ── DDP: read ranks early (torchrun sets these env vars) ────────────────────
# LOCAL_RANK  = rank within this node (0 to nproc_per_node-1)
# GLOBAL_RANK = rank across ALL nodes  (0 to world_size-1)
# IS_MAIN     = True only for the single global rank-0
LOCAL_RANK  = int(os.environ.get("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE  = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN     = GLOBAL_RANK == 0   # ← FIX: was LOCAL_RANK==0, caused 2 "mains" on multi-node

if IS_MAIN:
    print(f"✅ Cache directory set to: {HF_CACHE_DIR}")
    print(f"🌐 DDP — world_size={WORLD_SIZE}, local_rank={LOCAL_RANK}, global_rank={GLOBAL_RANK}")

import wandb
import random
import gc
import pandas as pd
from typing import Dict, Optional
from unsloth import FastLanguageModel, train_on_responses_only
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv


# ───────────────────────── UTILS ────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = CONFIG_PATH) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict):
    required = [
        "model.name", "training.output_dir", "wandb.project",
        "datasets",
    ]
    for path in required:
        keys = path.split(".")
        val = config
        for k in keys:
            val = val.get(k, {})
        if val == {}:
            raise ValueError(f"Missing config key: {path}")


# ───────────────────────── DATA PROCESSING ──────────────────────────────────

def sample_dataset(df: pd.DataFrame, sample_ratio: Optional[float] = None,
                   sample_n: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    if df.empty:
        return df
    if sample_n is not None:
        if sample_n <= 0:
            return pd.DataFrame(columns=df.columns)
        n = min(sample_n, len(df))
    elif sample_ratio is not None:
        if sample_ratio <= 0:
            return pd.DataFrame(columns=df.columns)
        if sample_ratio >= 1.0:
            return df.copy()
        n = int(len(df) * sample_ratio)
    else:
        return df.copy()
    if n >= len(df):
        return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def apply_chat_template_with_system(row, tokenizer, system_prompts: Dict[str, str], seed: int):
    messages = row["conversations"].copy()
    tools = row.get("tools", None)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
        tools=tools,
    )


def load_and_merge_datasets(config: Dict) -> pd.DataFrame:
    seed = config["training"]["seed"]
    all_rows = []
    seen_convos = set()

    for name, cfg in config["datasets"].items():
        if IS_MAIN:
            print(f"📥 Loading {name} from {cfg['path']}...")
        df = pd.read_json(cfg["path"], lines=True)
        df["_hash"] = df["conversations"].apply(
            lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        )
        before = len(df)
        df = df[~df["_hash"].isin(seen_convos)]
        after = len(df)
        if IS_MAIN:
            print(f"  → Removed {before - after} duplicates from prior datasets")

        sampled = sample_dataset(df,
                                 sample_ratio=cfg.get("sample_ratio"),
                                 sample_n=cfg.get("sample_n"),
                                 seed=seed)
        if IS_MAIN:
            print(f"  → Sampled {len(sampled):,} / {after:,} unique rows")

        seen_convos.update(sampled["_hash"])
        sampled = sampled.drop(columns=["_hash"])
        all_rows.append(sampled)

    if not all_rows:
        return pd.DataFrame()
    merged = pd.concat(all_rows, ignore_index=True)
    return merged.sample(frac=1, random_state=seed).reset_index(drop=True)

def process_to_dataset(config: Dict) -> tuple[Dataset, Optional[Dataset]]:
    seed    = config["training"]["seed"]
    max_len = config["model"]["max_seq_length"]

    cache_payload = {
        "datasets": config["datasets"],
        "max_seq_length": max_len,
        "model_name": config["model"]["name"],
    }
    cache_key   = hashlib.md5(
        json.dumps(cache_payload, sort_keys=True).encode()
    ).hexdigest()[:8]
    cache_base  = os.path.join(os.environ["HF_HOME"], f"proc_{cache_key}")
    train_cache = cache_base + "_train"
    eval_cache  = cache_base + "_eval"

    if IS_MAIN:
        if os.path.exists(train_cache):
            print(f"\n⚡ Dataset cache found, skipping processing.")
        else:
            print("\n🔧 Processing datasets (rank-0 only)...")

            tokenizer = AutoTokenizer.from_pretrained(
                config["model"]["name"],
                cache_dir=os.environ["HF_HOME"],
                use_fast=True,
            )

            # ── Load + deduplicate ────────────────────────────────────────
            df = load_and_merge_datasets(config)
            print(f"  Total merged rows: {len(df):,}")

            # ── Tokenize inline — one row at a time, no subprocess forking ─
            # This is the key fix: no num_proc, no multiprocessing,
            # no NCCL interference, no RAM spike from worker copies.
            print("💬 Applying chat templates + tokenizing...")
            input_ids_list      = []
            attention_mask_list = []
            skipped             = 0

            for i, row in enumerate(
                tqdm(df.itertuples(index=False), total=len(df), desc="Processing"),
                1
            ):
                try:
                    tools = getattr(row, "tools", None) or None
                    convs = getattr(row, "conversations")

                    text = tokenizer.apply_chat_template(
                        convs,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                        tools=tools,
                    )
                    enc = tokenizer(
                        text,
                        truncation=True,
                        max_length=max_len,
                        padding=False,
                        add_special_tokens=True,
                    )
                    input_ids_list.append(enc["input_ids"])
                    attention_mask_list.append(enc["attention_mask"])

                except Exception as e:
                    skipped += 1
                    if skipped <= 5:
                        print(f"  ⚠️  Row {i} failed: {e}")

                if i % 50_000 == 0:
                    print(f"  ... {i:,} / {len(df):,} tokenized")

            print(f"  ✅ Tokenized: {len(input_ids_list):,}  |  Skipped: {skipped:,}")
            del df; gc.collect()

            # ── Build Dataset — flat lists only, zero Arrow schema issues ──
            ds = Dataset.from_dict({
                "input_ids":      input_ids_list,
                "attention_mask": attention_mask_list,
            })
            del input_ids_list, attention_mask_list; gc.collect()

            # ── Split + save ───────────────────────────────────────────────
            eval_size = config["training"].get("eval_size", 0)
            if eval_size and eval_size > 0:
                split = ds.train_test_split(test_size=eval_size, seed=seed, shuffle=True)
                split["train"].save_to_disk(train_cache)
                split["test"].save_to_disk(eval_cache)
                print(f"  Train: {len(split['train']):,}  |  Eval: {len(split['test']):,}")
            else:
                ds.save_to_disk(train_cache)
                print(f"  Saved: {len(ds):,} samples")

            del ds; gc.collect()
            print("✅ Processing complete. Releasing barrier...")

    # ── All ranks sync here — rank-0 is guaranteed done writing ───────────
    if WORLD_SIZE > 1:
        dist.barrier()

    from datasets import load_from_disk
    train_ds = load_from_disk(train_cache)
    eval_ds  = load_from_disk(eval_cache) if os.path.exists(eval_cache) else None

    if IS_MAIN:
        print(f"  Train : {len(train_ds):,}")
        if eval_ds:
            print(f"  Eval  : {len(eval_ds):,}")

    return train_ds, eval_ds
# ───────────────────────── MODEL ────────────────────────────────────────────

def load_model(config: Dict):
    cache_dir = config.get("cache", {}).get("dir")
    if not cache_dir or cache_dir == "${HF_CACHE_DIR}":
        cache_dir = os.environ["HF_HOME"]

    if IS_MAIN:
        print("\n🧠 Loading model & tokenizer...")
    model_args = config["model"]
    lora_args  = config["lora"]

    model, processor = FastLanguageModel.from_pretrained(
        model_name=model_args["name"],
        max_seq_length=model_args["max_seq_length"],
        dtype=model_args["dtype"],
        load_in_4bit=model_args.get("load_in_4bit", False),
        fast_inference=False,   # vLLM-based, incompatible with DDP
        cache_dir=cache_dir,
    )
    tokenizer = processor.tokenizer

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_args["r"],
        target_modules=lora_args["target_modules"],
        lora_alpha=lora_args["alpha"],
        lora_dropout=lora_args["dropout"],
        bias=lora_args["bias"],
        use_gradient_checkpointing=lora_args["use_gradient_checkpointing"],
        random_state=config["training"]["seed"],
        use_rslora=lora_args["use_rslora"],
        loftq_config=lora_args["loftq_config"],
    )
    if IS_MAIN:
        print("✅ Model loaded.")
    return model, tokenizer


# ───────────────────────── TRAINING ─────────────────────────────────────────

def train(config: Dict):
    set_seed(config["training"]["seed"])

    # ── FIX: init DDP FIRST so dist.barrier() works inside process_to_dataset ─
    # Also ensures LOCAL_RANK is properly scoped to this node (0 to nproc-1),
    # preventing "invalid device ordinal" on nodes with fewer than 8 GPUs.
    if WORLD_SIZE > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=120),  # ← covers slow preprocessing
        )
        torch.cuda.set_device(LOCAL_RANK)
    output_dir = config["training"]["output_dir"]
    if IS_MAIN:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(CONFIG_PATH, os.path.join(output_dir, CONFIG_PATH))

    load_dotenv()
    if IS_MAIN:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config,
            mode="offline",
        )

    # Data — rank-0 processes, everyone else waits at barrier inside this fn
    train_ds, eval_ds = process_to_dataset(config)

    if IS_MAIN:
        print(f"Training samples  : {len(train_ds):,}")
        print(f"Evaluation samples: {len(eval_ds) if eval_ds else 0:,}")

    model, tokenizer = load_model(config)

    if IS_MAIN:
        print("\n🏃 Starting training...")
    tcfg = config["training"]

    training_args = SFTConfig(
        output_dir=tcfg["output_dir"],

        eval_strategy=tcfg.get("evaluation_strategy"),
        save_strategy=tcfg.get("save_strategy"),
        eval_steps=tcfg.get("eval_steps"),
        save_steps=tcfg.get("save_steps"),

        load_best_model_at_end=tcfg.get("load_best_model_at_end", False),
        metric_for_best_model=tcfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=tcfg.get("greater_is_better", False),

        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        num_train_epochs=tcfg["num_train_epochs"],
        logging_steps=tcfg["logging_steps"],
        optim=tcfg["optim"],

        max_length=config["model"]["max_seq_length"],
        packing=tcfg["packing"],
        save_total_limit=tcfg.get("save_total_limit"),
        log_level="info",

        report_to="wandb" if IS_MAIN else "none",
        run_name=config["wandb"]["run_name"],
        seed=config["training"]["seed"],

        dataset_text_field=None,
        dataloader_drop_last=False,
        dataset_num_proc=8,

        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )
    response_cfg = config["training"].get("response_format", {})
    instruction_part=response_cfg.get(
        "instruction_part",
        "<|im_start|>user\n"
    )
    response_part=response_cfg.get(
        "response_part",
        "<|im_start|>assistant\n"
    )  

    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )
    
    if IS_MAIN:
        example_answer = tokenizer.decode(
            [tokenizer.pad_token_id if x == -100 else x
             for x in trainer.train_dataset[0]["labels"]]
        ).replace(tokenizer.pad_token, " ")
        print("Example answer:\n", example_answer)

    trainer.train()

    if IS_MAIN:
        trainer.save_model(output_dir)
        print(f"\n📥 Model saved to: {output_dir}")

    del train_ds, eval_ds, model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    if IS_MAIN:
        wandb.finish()
        print("✅ Training completed.")

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


# ───────────────────────── MAIN ─────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config()
    validate_config(config)
    train(config)
