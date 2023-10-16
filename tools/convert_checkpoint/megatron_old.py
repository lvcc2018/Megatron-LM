# Merge the distributed LLaMA checkpoint into one
import torch
import json
from pathlib import Path
import os
import sys
import re
import types
import argparse
from collections import OrderedDict

from dataclasses import dataclass

def add_checkpointing_args(parser):
    parser.add_argument("--megatron_path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--param_size",
        type=int,
        required=True,
        default=30,
        help="Param size of the llama.",
    )
    parser.add_argument(
        "--source_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the source checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--source_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the source checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument("--print_checkpoint_structure", action="store_true")
    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp16",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=4,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    return parser

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 32000    # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-6

    max_batch_size: int = 32
    max_seq_len: int = 2048

config_dict = {
    7: ModelArgs(dim=4096, n_layers=32, n_heads=32),
    13: ModelArgs(dim=5120, n_layers=40, n_heads=40),
    30: ModelArgs(dim=6656, n_layers=60, n_heads=52),
    65: ModelArgs(dim=8192, n_layers=80, n_heads=64),
}

def merge_old_megatron(args):
    model_size = args.param_size
    config = config_dict[model_size]
    ffn_dim = config.multiple_of * \
        ((int(8*config.dim/3) + config.multiple_of-1) // config.multiple_of)
    config.ffn_dim = ffn_dim
    TP = args.source_tensor_model_parallel_size
    PP = args.source_pipeline_model_parallel_size
    load_path = args.load_path
    assert config.n_layers % PP == 0
    pp_num_layers = config.n_layers // PP
    TP_model_list = []
    # Load model and merge PP model
    saved_info = {}
    for tp in range(TP):
        current_tp_model = []
        for pp in range(PP):
            if PP == 1:
                model_dir_name = f"mp_rank_{tp:02d}"
            else:
                model_dir_name = f"mp_rank_{tp:02d}_{pp:03d}"
            model_path = os.path.join(load_path, model_dir_name, 'model_rng.pt')
            print(f"Load from {model_path}")
            current_model = torch.load(model_path, map_location='cpu')
            current_tp_model.append(current_model)
            if len(saved_info) == 0:
                for key in current_model:
                    if key != 'model':
                        saved_info[key] = current_model[key]
        if len(current_tp_model) > 1:
            new_tp_languge_model = OrderedDict()
            new_encoder = OrderedDict()
            for pp, pp_model in enumerate(current_tp_model):
                pp_encoder = pp_model['model']['language_model']['encoder']
                def replace_layer_idx(key_format, old_layer_idx, new_layer_idx):
                    new_key = key_format.format(new_layer_idx)
                    old_key = key_format.format(old_layer_idx)
                    new_encoder[new_key] = pp_encoder[old_key]
                if pp == 0:
                    new_encoder['freqs_cis'] = pp_encoder['freqs_cis']
                for layer_idx in range(pp_num_layers):
                    new_layer_idx = pp_num_layers * pp + layer_idx
                    input_layernorm_format = "layers.{}.input_layernorm.weight"
                    q_weight_layer_format = 'layers.{}.self_attention.query.weight'
                    k_weight_layer_format = 'layers.{}.self_attention.key.weight'
                    v_weight_layer_format = 'layers.{}.self_attention.value.weight'
                    dense_weight_layer_format = "layers.{}.self_attention.dense.weight"
                    post_layernorm_format = "layers.{}.post_attention_layernorm.weight"
                    mlp_h_to_4h_format = "layers.{}.mlp.dense_h_to_4h.weight"
                    mlp_dense_proj_format = "layers.{}.mlp.dence_proj.weight"
                    mlp_4h_to_h_format = "layers.{}.mlp.dense_4h_to_h.weight"
                    replace_layer_idx(input_layernorm_format, layer_idx, new_layer_idx)
                    replace_layer_idx(q_weight_layer_format, layer_idx, new_layer_idx)
                    replace_layer_idx(k_weight_layer_format, layer_idx, new_layer_idx)
                    replace_layer_idx(v_weight_layer_format, layer_idx, new_layer_idx)
                    replace_layer_idx(dense_weight_layer_format, layer_idx, new_layer_idx)
                    replace_layer_idx(post_layernorm_format, layer_idx, new_layer_idx)
                    replace_layer_idx(mlp_h_to_4h_format, layer_idx, new_layer_idx)
                    replace_layer_idx(mlp_dense_proj_format, layer_idx, new_layer_idx)
                    replace_layer_idx(mlp_4h_to_h_format, layer_idx, new_layer_idx)
                if pp == PP - 1: # last stage
                    new_encoder['final_layernorm.weight'] = pp_encoder['final_layernorm.weight']
            new_tp_languge_model['encoder'] = new_encoder
            new_tp_languge_model['embedding'] = current_tp_model[0]['model']['language_model']['embedding']
            new_tp_languge_model['lm_head'] = current_tp_model[-1]['model']['language_model']['lm_head']
            current_tp_model = new_tp_languge_model
        else:
            current_tp_model = current_tp_model[0]['model']['language_model']
        TP_model_list.append(current_tp_model)
    # merge TP model
    ## merge word embedding
    print(f"Other saved info {saved_info.keys()}")
    print("Merge different model to a whole model")
    word_embeddings = [m['embedding']['word_embeddings']['weight'] for m in TP_model_list]
    full_word_embedding = torch.cat(word_embeddings, dim=0)
    print("Embedding size", full_word_embedding.shape)
    # merge encoder
    encoders = [m['encoder'] for m in TP_model_list]
    full_encoder = OrderedDict()
    def concat_and_add_params(key, dim):
        full_encoder[key] = torch.cat([m[key] for m in encoders], dim=dim)
    full_encoder['freqs_cis'] = encoders[0]['freqs_cis']
    for layer_idx in range(config.n_layers):
        input_layernorm_name = f"layers.{layer_idx}.input_layernorm.weight"
        full_encoder[input_layernorm_name] = encoders[0][input_layernorm_name]
        q_weight_layer_name = f'layers.{layer_idx}.self_attention.query.weight'
        k_weight_layer_name = f'layers.{layer_idx}.self_attention.key.weight'
        v_weight_layer_name = f'layers.{layer_idx}.self_attention.value.weight'
        concat_and_add_params(q_weight_layer_name, dim=0)
        concat_and_add_params(k_weight_layer_name, dim=0)
        concat_and_add_params(v_weight_layer_name, dim=0)
        dense_weight_layer_name = f"layers.{layer_idx}.self_attention.dense.weight"
        concat_and_add_params(dense_weight_layer_name, dim=1)
        post_layernorm_name = f"layers.{layer_idx}.post_attention_layernorm.weight"
        full_encoder[post_layernorm_name] = encoders[0][post_layernorm_name]
        mlp_h_to_4h_name = f"layers.{layer_idx}.mlp.dense_h_to_4h.weight"
        mlp_dense_proj_name = f"layers.{layer_idx}.mlp.dence_proj.weight"
        mlp_4h_to_h_name = f"layers.{layer_idx}.mlp.dense_4h_to_h.weight"
        concat_and_add_params(mlp_h_to_4h_name, dim=0)
        concat_and_add_params(mlp_dense_proj_name, dim=0)
        concat_and_add_params(mlp_4h_to_h_name, dim=1)
    full_encoder['final_layernorm.weight'] = encoders[0]['final_layernorm.weight']
    lm_heads = [m['lm_head']['word_embeddings']['weight'] for m in TP_model_list]
    full_lm_head = torch.cat(lm_heads, dim=0)
    print("LM head size", full_word_embedding.shape)
    full_language_model = {
        'encoder': full_encoder,
        'embedding': {
            'word_embeddings': {
                'weight': full_word_embedding
            }
        },
        'lm_head': {
            'word_embeddings': {
                'weight': full_lm_head
            }
        }
    }
    return full_language_model

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
    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size(), val.dtype)
    else:
        print(msg, ":", val)

def convert_checkpoint_from_transformers_to_megatron(args):
    os.makedirs(args.save_path, exist_ok=True)
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.path.pardir)))
    sys.path.insert(0, args.megatron_path)
    try:
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    except ModuleNotFoundError:
        print("Unable to import Megatron. Exiting.")
        exit(1)
    checkpoint = merge_old_megatron(args)
    if args.print_checkpoint_structure:
        recursive_print("old", checkpoint)
    print("Successfully load the old ckpt. Start converting...")
    config = config_dict[args.param_size]
    ffn_dim = config.multiple_of * \
        ((int(8*config.dim/3) + config.multiple_of-1) // config.multiple_of)
    config.ffn_dim = ffn_dim
    config.max_position_embeddings=2048
    config.vocab_size = checkpoint['embedding']['word_embeddings']['weight'].shape[0]
    print("Model config:", config)
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
        "data_parallel_size": args.target_data_parallel_size,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "MixedTokenizer",
        "pad_vocab_size_to": None,
        "bias_gelu_fusion": False,
        "openai_gelu": False
    }
    print("Megatron Args:", megatron_args)
    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)
    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif  args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, "params_dtype", dtype)
    # save dummy optim state dict
    dummy_optim_state_dict = {}
    dummy_optim_state_dict["optimizer"] = {
        "step": 0,
        "param_groups": [
            {
                "lr": 0.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "eps": 0.0,
                "weight_decay": 0.0,
                "correct_bias": False,
                "params": [],
            }
        ],
    }
    if args.use_distributed_optimizer:
        for i in range(args.target_tensor_model_parallel_size):
            for j in range(args.target_pipeline_model_parallel_size):
                for k in range(args.target_data_parallel_size):
                    if args.target_pipeline_model_parallel_size== 1:
                        checkpoint_dir = f"mp_rank_{i:02d}_{k:03d}"
                    else:
                        checkpoint_dir = f"mp_rank_{i:02d}_{j:03d}_{k:03d}"
                    checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        dummy_optim_state_dict,
                        os.path.join(checkpoint_dir, "optim.pt"),
                    )
    # Convert.
    # print("Converting")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    # Embedding layer
    print("Converting embedding layer")
    word_embedding = checkpoint['embedding']['word_embeddings']['weight']
    output_embedding = checkpoint['lm_head']['word_embeddings']['weight']
    orig_vocab_size = word_embedding.shape[0]
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
        full_output_emb, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i].clone()
    # Transformer layers
    print("converting encoder layers")
    if config.n_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.n_layers}) must be divisible by number of tensor parallelism"
            f" ({args.target_pipeline_model_parallel_size})"
        )
    num_layers = config.n_layers // args.target_pipeline_model_parallel_size
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})
        
        def split_or_clone_weight(key_format, old_layer_idx, new_layer_idx, do_split, dim, do_cast=True, old_key=None, new_key=None):
            if old_key is None:
                old_key = key_format.format(old_layer_idx)
            if new_key is None:
                new_key = key_format.format(new_layer_idx)
            if do_cast:
                param = checkpoint['encoder'][old_key].to(dtype)
            else:
                param = checkpoint['encoder'][old_key]
            if do_split:
                params = torch.chunk(param, args.target_tensor_model_parallel_size, dim=dim)
                for i in range(args.target_tensor_model_parallel_size):
                    # split and clone in each process
                    index = i
                    params_dict = get_element_from_dict_by_path(
                        output_state_dict[index], "model.language_model.encoder")
                    params_dict[new_key] = params[i].clone()
            else:
                for i in range(args.target_tensor_model_parallel_size):
                    # clone in each process
                    params_dict = get_element_from_dict_by_path(
                        output_state_dict[i], "model.language_model.encoder")
                    params_dict[new_key] = param.clone()
        
        # split_or_clone_weight(
        #     key_format="freqs_cis",
        #     old_layer_idx=0,
        #     new_layer_idx=0,
        #     do_split=False,
        #     dim=0,
        #     do_cast=False,
        # )
        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            split_or_clone_weight(
                key_format="layers.{}.input_layernorm.weight",
                old_layer_idx=pp_layer_id,
                new_layer_idx=layer,
                do_split=False,
                dim=0,
            )
            split_or_clone_weight(
                key_format="layers.{}.self_attention.query.weight",
                old_layer_idx=pp_layer_id,
                new_layer_idx=layer,
                do_split=True,
                dim=0,
            )
            split_or_clone_weight(
                key_format="layers.{}.self_attention.key.weight",
                old_layer_idx=pp_layer_id,
                new_layer_idx=layer,
                do_split=True,
                dim=0,
            )
            split_or_clone_weight(
                key_format="layers.{}.self_attention.value.weight",
                old_layer_idx=pp_layer_id,
                new_layer_idx=layer,
                do_split=True,
                dim=0,
            )
            split_or_clone_weight(
                key_format="layers.{}.self_attention.dense.weight",
                old_layer_idx=pp_layer_id,
                new_layer_idx=layer,
                do_split=True,
                dim=1,
            )
            split_or_clone_weight(
                key_format="layers.{}.post_attention_layernorm.weight",
                old_layer_idx=pp_layer_id,
                new_layer_idx=layer,
                do_split=False,
                dim=1,
            )
            # Process mlp weights
            mlp_h_4h_name = f'layers.{layer}.mlp.dense_h_to_4h.weight'
            w1 = torch.chunk(checkpoint['encoder'][f'layers.{pp_layer_id}.mlp.dense_h_to_4h.weight'].to(
                dtype), args.target_tensor_model_parallel_size, dim=0)
            w3 = torch.chunk(checkpoint['encoder'][f'layers.{pp_layer_id}.mlp.dence_proj.weight'].to(
                dtype), args.target_tensor_model_parallel_size, dim=0)
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(
                    output_state_dict[i], "model.language_model.encoder")
                params_dict[mlp_h_4h_name] = torch.concat(
                    [w1[i], w3[i]], dim=0).clone()
            split_or_clone_weight(
                key_format="layers.{}.mlp.dense_4h_to_h.weight",
                old_layer_idx=pp_layer_id,
                new_layer_idx=layer,
                do_split=True,
                dim=1,
            )
        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            split_or_clone_weight(
                key_format="final_layernorm.weight",
                old_layer_idx=pp_layer_id,
                new_layer_idx=layer,
                do_split=False,
                dim=0,
            )
            # add the LM head
            for i in range(args.target_tensor_model_parallel_size):
                # Handle the LM head with the word embeddings?
                # params_dict = get_element_from_dict_by_path(
                #     output_state_dict[i], "model.word_embeddings_for_head")
                params_dict = get_element_from_dict_by_path(
                    output_state_dict[i], "model.language_model.output_layer")
                params_dict["weight"] = out_head_embed[i].clone()
        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            output_state_dict[tp_rank]["args"] = margs
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )
            if args.use_distributed_optimizer:
                checkpoint_name = "model_rng.pt"
            else:
                checkpoint_name = "model_optim_rng.pt"
                output_state_dict[tp_rank]["optimizer"] = dummy_optim_state_dict["optimizer"]
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(
                    f"Checkpoint structure of model state dict shard belonging to args.target_tensor_model_parallel_size rank {tp_rank} and args.target_pipeline_model_parallel_size rank"
                    f" {pp_rank}:"
                )
                recursive_print(f"tp {tp_rank}, pp {pp_rank}", output_state_dict[tp_rank])
            print(f"Save tp rank {tp_rank} pp rank {pp_rank} to {checkpoint_path}")
            torch.save(output_state_dict[tp_rank], checkpoint_path)    


if __name__ == '__main__':
    # merge_llama_ckpt()
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    args = parser.parse_args()
    convert_checkpoint_from_transformers_to_megatron(args)

#  python megatron_old.py --megatron_path /home/hyf/Megatron-LM --load_path /storage/huangyufei/pretrained_models/llama_chinese/llama_7b_WS16_TP8_PP1/iter_0018500 --save_path /storage/huangyufei/pretrained_models/llama_chinese/llama_7b_WS16_TP8_PP1_new --param_size 7 --source_tensor_model_parallel_size 8 --source_pipeline_model_parallel_size 1 --target_tensor_model_parallel_size 8 --target_pipeline_model_parallel_size 1 --target_data_parallel_size 4 --target_params_dtype bf16 --use_distributed_optimizer --print_checkpoint_structure
    



