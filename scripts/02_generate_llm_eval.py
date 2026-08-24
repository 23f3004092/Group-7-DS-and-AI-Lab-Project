import re
import json
import gc
import ctypes
import os
import argparse
import pandas as pd
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Reduces CUDA memory fragmentation — recommended by PyTorch for large model loading
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -- Paths --------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "data" / "processed" / "kcc"
IN_EVAL  = OUT_DIR / "kcc_eval_1.csv"
OUT_EVAL = OUT_DIR / "kcc_eval_1_augmented.csv"

# -- Config -------------------------------------------------------------------
LLM_MODEL_ID = "Qwen/Qwen3-8B"
BATCH_SIZE   = 8
LOG_INTERVAL = 8   # Print progress every N rows


# -- Helpers ------------------------------------------------------------------
def free_system_ram():
    """Force glibc to return freed heap pages to the OS immediately (Linux only)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # Windows dev machine -- gc.collect() is sufficient


def load_model(device="cuda:0"):
    print(f"Loading {LLM_MODEL_ID}...")
    torch.cuda.empty_cache()  # Clear any fragmented cache before loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        # device_map="auto" + max_memory: safe loading path (quantize on CPU first,
        # move 4-bit tensors to GPU) while forcing ALL layers onto GPU 0.
        # This eliminates the PCIe pipeline stall from 2-GPU split placement.
        device_map="auto",
        max_memory={0: "13GiB"},  # Force all layers to GPU 0; leave 3GB headroom for KV cache
        attn_implementation="sdpa",
    )
    free_system_ram()
    vram_info = ", ".join(
        f"GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f}GB"
        for i in range(torch.cuda.device_count())
    )
    print(f"Model loaded. VRAM: [{vram_info}]. System RAM freed.")
    return tokenizer, model


# -- Generation ---------------------------------------------------------------
def augment_eval(df, tokenizer, model, device="cuda:0"):
    """
    For each row, rewrites QueryText as a realistic farmer query (with noise)
    and generates one follow-up question. Saves both in a 3-turn JSON array.
    Language cycles: English -> Devanagari Hindi -> Hinglish (repeating).
    Thinking mode is disabled (enable_thinking=False) -- simple text
    transformation does not benefit from CoT reasoning.
    """
    print(f"Augmenting {len(df)} rows on {model.device}...")
    langs   = ["English", "Devanagari Hindi", "Hinglish (Hindi written in English alphabet)"]
    results = []
    # When device_map="auto", the model manages its own placement.
    # Move inputs to the model's primary device.
    input_device = model.device

    # Build all prompts first
    prompts = []
    for idx, (i, row) in enumerate(df.iterrows()):
        target_lang = langs[idx % 3]
        q = row['QueryText']
        a = row['KccAns']
        a_trimmed = str(a)[:300] + ("..." if len(str(a)) > 300 else "")

        system_prompt = (
            f"You are an expert data annotator specializing in agricultural dialects. "
            f"You MUST respond ONLY in {target_lang}. "
            f"Do NOT use any other language. "
            f"Output ONLY a valid JSON object with exactly two keys, and no extra text or markdown."
        )
        user_prompt = (
            f"Create a realistic farmer query and a follow-up question based on the Kisan Call Centre data below.\n\n"
            f"Original Query: {q}\n"
            f"Original Answer (summary): {a_trimmed}\n\n"
            f"Rules:\n"
            f"- Language: {target_lang} ONLY for all outputs.\n"
            f"- Augmented Query: Rewrite in first-person. Use simple rural vocabulary. "
            f"Insert occasional disfluencies or minor spelling errors typical of a transcribed phone call.\n"
            f"- Follow-up Question: Exactly ONE logical follow-up the farmer would ask after hearing the answer.\n"
            f"Output ONLY this JSON, nothing else:\n"
            "{{\n"
            "  \"augmented_query\": \"<rewritten query with noise>\",\n"
            "  \"follow_up_question\": \"<the single follow up question>\"\n"
            "}}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append((i, row, text, target_lang))

    # Batched generation
    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch       = prompts[batch_start: batch_start + BATCH_SIZE]
        batch_texts = [b[2] for b in batch]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(input_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                # max_new_tokens=128: JSON output is ~100 tokens.
                # Reducing from 256 prevents a single long sequence from stalling
                # the entire batch (all seqs wait for the slowest one).
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, b in enumerate(batch):
            row         = b[1].copy()
            target_lang = b[3]
            q           = row['QueryText']
            a           = row['KccAns']

            text_out = tokenizer.decode(
                outputs[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            text_out = re.sub(r'<think>.*?</think>', '', text_out, flags=re.DOTALL).strip()

            try:
                json_str = re.search(r'\{.*\}', text_out, re.DOTALL).group(0)
                parsed   = json.loads(json_str)
                row['multi_turn_json'] = json.dumps([
                    {"role": "user",      "content": parsed.get("augmented_query",    q)},
                    {"role": "assistant", "content": str(a)},
                    {"role": "user",      "content": parsed.get("follow_up_question", "")},
                ], ensure_ascii=False)
            except Exception:
                row['multi_turn_json'] = json.dumps([
                    {"role": "user",      "content": f"(Rephrased to {target_lang}) {q}"},
                    {"role": "assistant", "content": str(a)},
                ], ensure_ascii=False)

            row['language'] = target_lang
            results.append(row)

        rows_done = min(batch_start + BATCH_SIZE, len(prompts))
        if rows_done % LOG_INTERVAL == 0 or rows_done == len(prompts):
            print(f"  [{rows_done}/{len(prompts)}] rows processed")
            try:
                print(f"  Sample: {results[-1]['multi_turn_json'][:120]}...")
            except UnicodeEncodeError:
                print("  Sample: [Unicode content hidden]")

    return pd.DataFrame(results)


# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM Eval Augmentation (single GPU)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip generation (for testing)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"02_generate_llm_eval.py  |  device={device}")
    print("=" * 60)

    if not IN_EVAL.exists():
        raise FileNotFoundError(f"Missing {IN_EVAL}. Run 01_prepare_datasets.py first.")

    eval_df = pd.read_csv(IN_EVAL)
    print(f"Loaded {len(eval_df)} rows from {IN_EVAL}")

    if not args.skip_llm:
        tokenizer, model = load_model(device)
        eval_df = augment_eval(eval_df, tokenizer, model, device)
    else:
        print("--skip-llm: skipping generation.")
        eval_df['multi_turn_json'] = "[]"
        eval_df['language']        = "English"

    eval_df.to_csv(OUT_EVAL, index=False)
    print(f"\nSaved {len(eval_df)} rows -> {OUT_EVAL}")
    print("Done.")


if __name__ == "__main__":
    main()
