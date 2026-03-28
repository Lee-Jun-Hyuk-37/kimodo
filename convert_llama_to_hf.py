"""Convert Meta Llama 3-8B-Instruct checkpoint to HuggingFace format.

Uses the exact key mapping from the official transformers conversion script,
but saves directly via safetensors without loading through LlamaForCausalLM
to avoid version compatibility issues.

Verified against: transformers convert_llama_weights_to_hf.py (main branch)
"""
import json
import os
import shutil

import torch
from safetensors.torch import save_file


def read_json(path):
    with open(path) as f:
        return json.load(f)


def permute(w, n_heads, dim1, dim2):
    """Permute attention weights from Meta format to HF format."""
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert():
    input_dir = os.path.expanduser("~/.llama/checkpoints/Meta-Llama-3-8B-Instruct")
    output_dir = os.path.expanduser("~/.llama/hf/Meta-Llama-3-8B-Instruct")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    params = read_json(os.path.join(input_dir, "params.json"))
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_kv_heads = params.get("n_kv_heads", n_heads)
    dim = params["dim"]
    dims_per_head = dim // n_heads

    print(f"Model: {n_layers} layers, {n_heads} heads, {n_kv_heads} kv_heads, dim={dim}")

    print("Loading Meta checkpoint...")
    state_dict = torch.load(
        os.path.join(input_dir, "consolidated.00.pth"),
        map_location="cpu",
        weights_only=True,
    )
    print(f"Loaded {len(state_dict)} tensors")

    # Key mapping with attention weight permutation (critical for correctness)
    hf_dict = {}
    for key, value in state_dict.items():
        if key == "tok_embeddings.weight":
            hf_dict["model.embed_tokens.weight"] = value
        elif key == "norm.weight":
            hf_dict["model.norm.weight"] = value
        elif key == "output.weight":
            hf_dict["lm_head.weight"] = value
        elif key.startswith("layers."):
            parts = key.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])

            if rest == "attention.wq.weight":
                hf_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = permute(
                    value, n_heads, dim, dim
                )
            elif rest == "attention.wk.weight":
                hf_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = permute(
                    value, n_kv_heads, n_kv_heads * dims_per_head, dim
                )
            elif rest == "attention.wv.weight":
                hf_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = value
            elif rest == "attention.wo.weight":
                hf_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = value
            elif rest == "feed_forward.w1.weight":
                hf_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = value
            elif rest == "feed_forward.w2.weight":
                hf_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = value
            elif rest == "feed_forward.w3.weight":
                hf_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = value
            elif rest == "attention_norm.weight":
                hf_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = value
            elif rest == "ffn_norm.weight":
                hf_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = value
            else:
                print(f"  WARNING: unmapped key {key}")
        else:
            print(f"  WARNING: unmapped key {key}")

    print(f"Mapped {len(hf_dict)} tensors")

    # Save weights
    print("Saving model.safetensors...")
    save_file(hf_dict, os.path.join(output_dir, "model.safetensors"))

    # Write config.json (matches HF meta-llama/Meta-Llama-3-8B-Instruct exactly)
    intermediate_size = 14336  # Llama 3 8B specific
    config = {
        "_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": dim,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": 8192,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": n_heads,
        "num_hidden_layers": n_layers,
        "num_key_value_heads": n_kv_heads,
        "pretraining_tp": 1,
        "rms_norm_eps": params["norm_eps"],
        "rope_scaling": None,
        "rope_theta": params.get("rope_theta", 500000.0),
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "5.1.0",
        "use_cache": True,
        "vocab_size": params["vocab_size"],
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer from McGill-NLP (already in HF format, public)
    print("Downloading tokenizer from McGill-NLP...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
    tokenizer.save_pretrained(output_dir)

    print(f"\nConversion complete: {output_dir}")
    files = os.listdir(output_dir)
    for f in sorted(files):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f}: {size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"  {f}: {size / 1024:.1f} KB")


if __name__ == "__main__":
    convert()
