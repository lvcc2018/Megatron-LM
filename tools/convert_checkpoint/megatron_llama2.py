# Merge the distributed LLaMA checkpoint into one
import torch
import json
from pathlib import Path
import os
import sys
import re
import types
import argparse
from params_dict import megatron_to_llama2, tensor_parallel_params
from typing import Optional

from dataclasses import dataclass


def add_checkpointing_args(parser):
    parser.add_argument(
        "--megatron-path",
        type=str,
        default=None,
        help="Base directory of Megatron repository."
    )
    parser.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--addition-vocab-size",
        type=int,
        default=26276,
        help="Additional word embedding size.",
    )
    parser.add_argument(
        "--print-checkpoint-structure",
        action="store_true"
    )
    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="bf16",
        help=(
            "The dtype of the converted checkpoint. "
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=1,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
        ),
    )
    parser.add_argument(
        "--num_query_groups",
        type=int,
        default=8
    )
    parser.add_argument(
        '--no-merge-qkv', 
        action='store_false', 
        default=True,
        help='Split the query/key/value matrix.',
        dest='merge_qkv'
    )
    return parser


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048


llama_to_megatron = {v[1:-1]: k for k,
                     v in megatron_to_llama2.items()}


def merge_llama_ckpt(args):
    ckpt_save_path = os.path.join(args.load_path, f'checkpoints.pt')
    if os.path.exists(ckpt_save_path):
        print("Checkpoint exists, loading...")
        return torch.load(ckpt_save_path)
    print("Checkpoint not exists, merging...")
    checkpoints = sorted(
        Path(args.load_path).glob("*.pth"))
    checkpoints = [torch.load(i, map_location='cpu') for i in checkpoints]

    with open(Path(args.load_path) / "params.json", "r") as f:
            params = json.loads(f.read())
    config: ModelArgs = ModelArgs(
        max_seq_len=4096,
        **params,
    )
    ffn_dim = 4 * config.dim
    ffn_dim = int(2 * ffn_dim / 3)
    if config.ffn_dim_multiplier is not None:
        ffn_dim = int(config.ffn_dim_multiplier * ffn_dim)
    ffn_dim = config.multiple_of * ((ffn_dim + config.multiple_of - 1) // config.multiple_of)
    config.ffn_dim = ffn_dim
    config.max_position_embeddings = 4096
    config.vocab_size = 32000
    print(config)
    
    res = {}
    res['tok_embeddings.weight'] = torch.cat(
        [checkpoints[i]['tok_embeddings.weight'] for i in range(len(checkpoints))], dim=1).clone()
    res['norm.weight'] = checkpoints[0]['norm.weight'].clone()
    res['output.weight'] = torch.cat(
        [checkpoints[i]['output.weight'] for i in range(len(checkpoints))], dim=0).clone()
    assert res['tok_embeddings.weight'].shape == torch.Size([config.vocab_size, config.dim])
    assert res['output.weight'].shape == torch.Size([config.vocab_size, config.dim])
    '''
    'layers.0.attention.wq.weight'
    'layers.0.attention.wk.weight'
    'layers.0.attention.wv.weight'
    'layers.0.attention.wo.weight'
    'layers.0.feed_forward.w1.weight'
    'layers.0.feed_forward.w2.weight'
    'layers.0.feed_forward.w3.weight'
    'layers.0.attention_norm.weight'
    'layers.0.ffn_norm.weight',
    '''
    for i in range(config.n_layers):
        for j in ['q', 'k', 'v', 'o']:
            res[f'layers.{str(i)}.attention.w{j}.weight'] = torch.cat(
                [
                    checkpoints[k][f'layers.{str(i)}.attention.w{j}.weight'] 
                        for k in range(len(checkpoints))
                ], dim=int(j == 'o')
            ).clone()
        for j in range(1, 4):
            res[f'layers.{str(i)}.feed_forward.w{str(j)}.weight'] = torch.cat(
                [
                    checkpoints[k][f'layers.{str(i)}.feed_forward.w{str(j)}.weight'] 
                        for k in range(len(checkpoints))
                ], dim=int(j == 2)).clone()
        res[f'layers.{str(i)}.attention_norm.weight'] = checkpoints[0][f'layers.{str(i)}.attention_norm.weight'].clone()
        res[f'layers.{str(i)}.ffn_norm.weight'] = checkpoints[0][f'layers.{str(i)}.ffn_norm.weight'].clone()
    torch.save(res, ckpt_save_path)
    return res


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint.

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def convert_checkpoint_from_llama2_to_megatron(args):
    os.makedirs(args.save_path, exist_ok=True)
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.path.pardir)))
    sys.path.insert(0, args.megatron_path)

    try:
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    except ModuleNotFoundError:
        print("Unable to import Megatron. Exiting.")
        exit(1)
    checkpoint = merge_llama_ckpt(args)
    print("Successfully load the llama ckpt. Start converting...")

    params = json.loads(
        open(os.path.join(args.load_path,f'params.json'), 'r').read())
    config: ModelArgs = ModelArgs(
        max_seq_len=4096,
        **params,
    )
    ffn_dim = 4 * config.dim
    ffn_dim = int(2 * ffn_dim / 3)
    if config.ffn_dim_multiplier is not None:
        ffn_dim = int(config.ffn_dim_multiplier * ffn_dim)
    ffn_dim = config.multiple_of * ((ffn_dim + config.multiple_of - 1) // config.multiple_of)
    config.ffn_dim = ffn_dim
    config.max_position_embeddings = 4096
    config.vocab_size = 32000
    print(config)

    # Saving the tracker file
    tracker_filepath = os.path.join(
        args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # megatron args
    megatron_args = {
        "orig_vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size": config.dim,
        "num_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "ffn_hidden_size": config.ffn_dim,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
        "data_parallel_size": 1,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "SentencePieceTokenizer",
        "pad_vocab_size_to": None,
        "bias_gelu_fusion": False,
        "openai_gelu": False
    }
    print(megatron_args)

    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, "params_dtype", dtype)

    # Convert.
    # print("Converting")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    # Embedding layer
    # 获取Embedding Layer
    print("Converting embedding layer")
    word_embedding = checkpoint['tok_embeddings.weight']
    output_embedding = checkpoint['output.weight']
    if args.addition_vocab_size != 0:
        addition_word_embedding = get_initialized_word_embeddings(
            mean=torch.mean(word_embedding),
            std=torch.std(word_embedding),
            vocab_size=args.addition_vocab_size,
            hidden_size=config.dim,
            dtype=dtype,
        )
        addition_lm_head = get_initialized_word_embeddings(
            mean=torch.mean(output_embedding),
            std=torch.std(output_embedding),
            vocab_size=args.addition_vocab_size,
            hidden_size=config.dim,
            dtype=dtype,
        )
        word_embedding = torch.cat(
            (word_embedding, addition_word_embedding), dim=0)
        output_embedding = torch.cat(
            (output_embedding, addition_lm_head), dim=0)
    orig_vocab_size = config.vocab_size + args.addition_vocab_size
    assert word_embedding.shape[0] == orig_vocab_size
    assert output_embedding.shape[0] == orig_vocab_size
    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
    setattr(margs, "padded_vocab_size", padded_vocab_size)
    if orig_vocab_size > padded_vocab_size:
        full_word_embed = word_embedding[0:padded_vocab_size, :]
        full_output_emb = output_embedding[0:padded_vocab_size, :]
    elif orig_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - orig_vocab_size
        full_word_embed = torch.cat(
            (word_embedding, word_embedding[-1].unsqueeze(0).expand(padding_size, -1)))
        full_output_emb = torch.cat(
            (output_embedding, output_embedding[-1].unsqueeze(0).expand(padding_size, -1)))
    else:
        full_word_embed = word_embedding
        full_output_emb = output_embedding

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(
        full_word_embed, args.target_tensor_model_parallel_size, dim=0)
    out_head_embed = torch.chunk(
        full_output_emb, args.target_tensor_model_parallel_size, dim=0
    )
    for i in range(args.target_tensor_model_parallel_size):
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i].clone()

    # Transformer layers
    print("converting transformer layers")
    if config.n_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.n_layers}) must be divisible by number of tensor parallelism"
            f" ({args.target_pipeline_model_parallel_size})"
        )
    num_layers = config.n_layers // args.target_pipeline_model_parallel_size

    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})
        # rope_freqs = checkpoint['rope.freqs']

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in checkpoint.keys()
                if layer_name.startswith(f"layers.{pp_layer_id}.")
            ]

            # Process query, key, value weights
            q_layer_weight = torch.chunk(checkpoint[f'layers.{pp_layer_id}.attention.wq.weight'].to(
                dtype), args.target_tensor_model_parallel_size, dim=0)
            k_layer_weight = torch.chunk(checkpoint[f'layers.{pp_layer_id}.attention.wk.weight'].to(
                dtype), args.target_tensor_model_parallel_size, dim=0)
            v_layer_weight = torch.chunk(checkpoint[f'layers.{pp_layer_id}.attention.wv.weight'].to(
                dtype), args.target_tensor_model_parallel_size, dim=0)
            if args.merge_qkv:
                qkv_weight_layer_name = f'layers.{layer}.self_attention.query_key_value.weight'
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(
                        output_state_dict[i], "model.language_model.encoder")
                    num_query_groups_per_partition = args.num_query_groups // args.target_tensor_model_parallel_size
                    q_layer_weight_partition = torch.chunk(
                        q_layer_weight[i],
                        num_query_groups_per_partition,
                        dim=0)
                    k_layer_weight_partition = torch.chunk(
                        k_layer_weight[i],
                        num_query_groups_per_partition,
                        dim=0)
                    v_layer_weight_partition = torch.chunk(
                        v_layer_weight[i],
                        num_query_groups_per_partition,
                        dim=0)
                    cat_list = []
                    for j in range(num_query_groups_per_partition):
                        cat_list.append(q_layer_weight_partition[j])
                        cat_list.append(k_layer_weight_partition[j])
                        cat_list.append(v_layer_weight_partition[j])
                    params_dict[qkv_weight_layer_name] = torch.cat(
                        cat_list, dim=0).clone()
            else:
                q_weight_layer_name = f'layers.{layer}.self_attention.query.weight'
                k_weight_layer_name = f'layers.{layer}.self_attention.key.weight'
                v_weight_layer_name = f'layers.{layer}.self_attention.value.weight'
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(
                        output_state_dict[i], "model.language_model.encoder")
                    params_dict[q_weight_layer_name] = q_layer_weight[i].clone()
                    params_dict[k_weight_layer_name] = k_layer_weight[i].clone()
                    params_dict[v_weight_layer_name] = v_layer_weight[i].clone()

            # Process mlp weights
            mlp_h_4h_name = f'layers.{layer}.mlp.dense_h_to_4h.weight'
            mlp_4h_h_name = f'layers.{layer}.mlp.dense_4h_to_h.weight'
            w1 = torch.chunk(checkpoint[f'layers.{pp_layer_id}.feed_forward.w1.weight'].to(
                dtype), args.target_tensor_model_parallel_size, dim=0)
            w2 = torch.chunk(checkpoint[f'layers.{pp_layer_id}.feed_forward.w2.weight'].to(
                dtype), args.target_tensor_model_parallel_size, dim=1)
            w3 = torch.chunk(checkpoint[f'layers.{pp_layer_id}.feed_forward.w3.weight'].to(
                dtype), args.target_tensor_model_parallel_size, dim=0)
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(
                    output_state_dict[i], "model.language_model.encoder")
                params_dict[mlp_h_4h_name] = torch.concat(
                    [w1[i], w3[i]], dim=0).clone()
                params_dict[mlp_4h_h_name] = w2[i].clone()

            # Others
            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                if m is None:
                    break

                # The index of the layer.
                _ = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight_or_bias = m.group(3)

                params = checkpoint[layer_name].to(dtype)
                # handle layernorm
                if op_name.endswith("norm"):
                    out_name = "input_layernorm" if op_name.startswith(
                        "attention") else "post_attention_layernorm"
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

                # handle attention proj weights
                elif op_name.startswith("attention.wo") and weight_or_bias == "weight":
                    out_name = llama_to_megatron.get(op_name, None)
                    if out_name is None:
                        continue
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"
                # QKV have been processed
                elif op_name.startswith("attention."):
                    continue

                # handle attention and mlp weights
                # attention and mlp weights have been processed
                elif weight_or_bias == "weight":
                    continue

                # skip
                else:
                    print(f'Unknown layer: {layer_name}')
                    continue

                if op_name + "." + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name == "attention.wo" else 0
                    params = torch.chunk(
                        params, args.target_tensor_model_parallel_size, dim=dim)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(
                        output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i].clone() if (
                            op_name + "." + weight_or_bias in tensor_parallel_params) else params.clone()
                    )

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            for weight_or_bias in ["weight"]:
                params = checkpoint[f"norm.{weight_or_bias}"].to(
                    dtype)
                layer_name = f"final_layernorm.{weight_or_bias}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(
                        output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = params.clone()

            # add the LM head
            for i in range(args.target_tensor_model_parallel_size):
                # params_dict = get_element_from_dict_by_path(
                #     output_state_dict[i], "model.word_embeddings_for_head")
                params_dict = get_element_from_dict_by_path(
                    output_state_dict[i], "model.language_model.output_layer")
                params_dict["weight"] = out_head_embed[i].clone()

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )
            checkpoint_name = "model_optim_rng.pt"
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(
                    f"Checkpoint structure of model state dict shard belonging to args.target_tensor_model_parallel_size rank {tp_rank} and args.target_pipeline_model_parallel_size rank"
                    f" {pp_rank}:"
                )
                recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)


def get_initialized_word_embeddings(mean, std, vocab_size, hidden_size, dtype=torch.float32, cache=None):
    """Get initialized word embeddings.

    Args:
        mean (float): mean of the normal distribution.
        std (float): standard deviation of the normal distribution.
        vocab_size (int): vocabulary size.
        hidden_size (int): hidden size.
        dtype (torch.dtype, optional): data type. Defaults to torch.float32.

    Returns:
        torch.Tensor: initialized word embeddings.
    """
    if cache is not None:
        cached_word_embedding = torch.load(
            cache, map_location='cpu', dtype=dtype)
        assert cached_word_embedding.size() == (vocab_size, hidden_size)
        return cached_word_embedding
    return torch.normal(mean=mean, std=std, size=(vocab_size, hidden_size), dtype=dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    args = parser.parse_args()
    convert_checkpoint_from_llama2_to_megatron(args)
