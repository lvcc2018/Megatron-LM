import torch
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir), os.path.pardir)))
from collections import OrderedDict
import copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model-dir", type=str, help='Model path for converting.')
    parser.add_argument("--input-tp", type=int)
    parser.add_argument("--input-pp", type=int)
    parser.add_argument("--target-model-dir", type=str)
    parser.add_argument("--target-tp", type=int)
    parser.add_argument("--target-dp", type=int)
    parser.add_argument("--target-pp", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--merge_qkv", action="store_true")
    parser.add_argument("--original-vocab-size", type=int, default=None)
    parser.add_argument("--extra-vocab-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    TP = args.input_tp
    PP = args.input_pp
    pp_num_layers = args.num_layers // PP
    TP_model_list = []
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    # Load model and merge PP model
    saved_info = {}
    for tp in range(TP):
        current_tp_model = []
        for pp in range(PP):
            if PP == 1:
                model_dir_name = f"mp_rank_{tp:02d}"
            else:
                model_dir_name = f"mp_rank_{tp:02d}_{pp:03d}"
            model_path = os.path.join(args.input_model_dir, model_dir_name, 'model_rng.pt')
            if not os.path.isfile(model_path):
                print("use model_optim_rng.pt")
                model_path = os.path.join(args.input_model_dir, model_dir_name, 'model_optim_rng.pt')
            print(f"Load from {model_path}")
            current_model = torch.load(model_path, map_location='cpu')
            current_tp_model.append(current_model)
            if len(saved_info) == 0:
                for key in current_model:
                    if key != 'model':
                        saved_info[key] = current_model[key]
                saved_args = saved_info['args']
                saved_args.tensor_model_parallel_size = args.target_tp
                saved_args.pipeline_model_parallel_size = args.target_pp
                saved_args.data_parallel_size = args.target_dp
        if len(current_tp_model) > 1:
            new_tp_languge_model = OrderedDict()
            new_encoder = OrderedDict()
            for pp, pp_model in enumerate(current_tp_model):
                pp_encoder = pp_model['model']['language_model']['encoder']
                def replace_layer_idx(key_format, old_layer_idx, new_layer_idx):
                    new_key = key_format.format(new_layer_idx)
                    old_key = key_format.format(old_layer_idx)
                    new_encoder[new_key] = pp_encoder[old_key]
                # if pp == 0:
                #     new_encoder['freqs_cis'] = pp_encoder['freqs_cis']
                for layer_idx in range(pp_num_layers):
                    new_layer_idx = pp_num_layers * pp + layer_idx
                    input_layernorm_format = "layers.{}.input_layernorm.weight"
                    if args.merge_qkv:
                        qkv_weight_layer_format = 'layers.{}.self_attention.query_key_value.weight'
                    else:
                        q_weight_layer_format = 'layers.{}.self_attention.query.weight'
                        k_weight_layer_format = 'layers.{}.self_attention.key.weight'
                        v_weight_layer_format = 'layers.{}.self_attention.value.weight'

                    dense_weight_layer_format = "layers.{}.self_attention.dense.weight"
                    post_layernorm_format = "layers.{}.post_attention_layernorm.weight"
                    mlp_h_to_4h_format = "layers.{}.mlp.dense_h_to_4h.weight"
                    # mlp_dense_proj_format = "layers.{}.mlp.dence_proj.weight"
                    mlp_4h_to_h_format = "layers.{}.mlp.dense_4h_to_h.weight"
                    replace_layer_idx(input_layernorm_format, layer_idx, new_layer_idx)
                    if args.merge_qkv:
                        replace_layer_idx(qkv_weight_layer_format, layer_idx, new_layer_idx)
                    else:
                        replace_layer_idx(q_weight_layer_format, layer_idx, new_layer_idx)
                        replace_layer_idx(k_weight_layer_format, layer_idx, new_layer_idx)
                        replace_layer_idx(v_weight_layer_format, layer_idx, new_layer_idx)
                    replace_layer_idx(dense_weight_layer_format, layer_idx, new_layer_idx)
                    replace_layer_idx(post_layernorm_format, layer_idx, new_layer_idx)
                    replace_layer_idx(mlp_h_to_4h_format, layer_idx, new_layer_idx)
                    # replace_layer_idx(mlp_dense_proj_format, layer_idx, new_layer_idx)
                    replace_layer_idx(mlp_4h_to_h_format, layer_idx, new_layer_idx)
                if pp == PP - 1: # last stage
                    new_encoder['final_layernorm.weight'] = pp_encoder['final_layernorm.weight']
            new_tp_languge_model['encoder'] = new_encoder
            new_tp_languge_model['embedding'] = current_tp_model[0]['model']['language_model']['embedding']
            new_tp_languge_model['output_layer'] = current_tp_model[-1]['model']['language_model']['output_layer']
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
    if args.original_vocab_size is not None:
        full_word_embedding = full_word_embedding[:args.original_vocab_size, :]  # release pad vocab
    print("Embedding size", full_word_embedding.shape)
    # merge encoder
    encoders = [m['encoder'] for m in TP_model_list]
    full_encoder = OrderedDict()
    def concat_and_add_params(key, dim):
        full_encoder[key] = torch.cat([m[key] for m in encoders], dim=dim)
    # full_encoder['freqs_cis'] = encoders[0]['freqs_cis']
    for layer_idx in range(args.num_layers):
        input_layernorm_name = f"layers.{layer_idx}.input_layernorm.weight"
        full_encoder[input_layernorm_name] = encoders[0][input_layernorm_name]
        if args.merge_qkv:
            qkv_weigth_layer_name = f'layers.{layer_idx}.self_attention.query_key_value.weight'
            concat_and_add_params(qkv_weigth_layer_name, dim=0)
        else:
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
        # mlp_dense_proj_name = f"layers.{layer_idx}.mlp.dence_proj.weight"
        mlp_4h_to_h_name = f"layers.{layer_idx}.mlp.dense_4h_to_h.weight"
        # concat_and_add_params(mlp_h_to_4h_name, dim=0)
        mlp_h_to_4h_params = [torch.chunk(m[mlp_h_to_4h_name], 2, dim=0) for m in encoders]
        full_encoder[mlp_h_to_4h_name] = torch.cat([m[0] for m in mlp_h_to_4h_params] + [m[1] for m in mlp_h_to_4h_params], dim=0)
        # concat_and_add_params(mlp_dense_proj_name, dim=0)
        concat_and_add_params(mlp_4h_to_h_name, dim=1)
    full_encoder['final_layernorm.weight'] = encoders[0]['final_layernorm.weight']
    lm_heads = [m['output_layer']['weight'] for m in TP_model_list]
    full_lm_head = torch.cat(lm_heads, dim=0)
    if args.original_vocab_size is not None:
        full_lm_head = full_lm_head[:args.original_vocab_size, :]  # release pad vocab
    # for key in full_encoder:
    #     print(key, full_encoder[key].shape, full_encoder[key].dtype)
    # quit()
    if args.extra_vocab_size is not None:
        print(f"Add {args.extra_vocab_size} extra tokens")
        with torch.no_grad():
            hidden_size = full_word_embedding.shape[1]
            word_embeddings_mean = torch.mean(full_word_embedding, dim=0, keepdim=True)
            word_embeddings_std = torch.std(full_word_embedding, dim=0, keepdim=True)
            word_embeddings_mean = word_embeddings_mean.repeat(args.extra_vocab_size, 1)
            word_embeddings_std = word_embeddings_std.repeat(args.extra_vocab_size, 1)
            extra_word_embeddings = torch.normal(word_embeddings_mean, word_embeddings_std, generator=generator)
            lm_head_mean = torch.mean(full_lm_head, dim=0, keepdim=True)
            lm_head_std = torch.std(full_lm_head, dim=0, keepdim=True)
            lm_head_mean = lm_head_mean.repeat(args.extra_vocab_size, 1)
            lm_head_std = lm_head_std.repeat(args.extra_vocab_size, 1)
            extra_lm_head = torch.normal(lm_head_mean, lm_head_std, generator=generator)
            full_word_embedding = torch.cat([full_word_embedding, extra_word_embeddings], dim=0)
            full_lm_head = torch.cat([full_lm_head, extra_lm_head], dim=0)
            print(f"New vocab size: {full_word_embedding.shape}")
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
    # split to new model
    TP = args.target_tp
    PP = args.target_pp
    DP = args.target_dp
    print(f"Split the model to tensor parallel {TP}, pipeline parallel {PP}, data parallel {DP}")
    print("converting encoder layers")
    if args.num_layers % args.target_pp != 0:
        raise ValueError(
            f"Number of layers ({args.num_layers}) must be divisible by number of tensor parallelism"
            f" ({PP})"
        )
    current_vocab_size = full_word_embedding.shape[0]
    assert current_vocab_size % TP == 0, f"Vocab size {current_vocab_size} is not divisible by tensor parallel size" # TODO: make pad from vocab
    new_tp_word_embeddings = torch.chunk(full_word_embedding, TP, dim=0)
    new_tp_lm_heads = torch.chunk(full_lm_head, TP, dim=0)
    num_layers = args.num_layers // PP
    iter_name = os.path.basename(args.input_model_dir)
    os.makedirs(args.target_model_dir, exist_ok=True)
    output_dir = os.path.join(args.target_model_dir, iter_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Save new model to {output_dir}")
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
    for pp_rank in range(PP):
        new_tp_encoders = [OrderedDict() for _ in range(TP)]
        def split_and_add_params(key, old_key, dim):
            original_param = full_encoder[old_key]
            splited_params = torch.chunk(original_param, TP, dim=dim)
            for encoder, param in zip(new_tp_encoders, splited_params):
                encoder[key] = param.clone()
        for layer_idx in range(num_layers):
            pp_layer_idx = pp_rank * num_layers + layer_idx
            input_layernorm_name = f"layers.{layer_idx}.input_layernorm.weight"
            old_input_layernorm_name = f"layers.{pp_layer_idx}.input_layernorm.weight"
            for encoder in new_tp_encoders:
                encoder[input_layernorm_name] = full_encoder[old_input_layernorm_name].clone()
            if args.merge_qkv:
                qkv_weight_layer_name = f'layers.{layer_idx}.self_attention.query_key_value.weight'
                old_qkv_weight_layer_name = f'layers.{pp_layer_idx}.self_attention.query_key_value.weight'
                split_and_add_params(qkv_weight_layer_name, old_qkv_weight_layer_name, dim=0)

            else:
                q_weight_layer_name = f'layers.{layer_idx}.self_attention.query.weight'
                k_weight_layer_name = f'layers.{layer_idx}.self_attention.key.weight'
                v_weight_layer_name = f'layers.{layer_idx}.self_attention.value.weight'
                old_q_weight_layer_name = f'layers.{pp_layer_idx}.self_attention.query.weight'
                old_k_weight_layer_name = f'layers.{pp_layer_idx}.self_attention.key.weight'
                old_v_weight_layer_name = f'layers.{pp_layer_idx}.self_attention.value.weight'
                split_and_add_params(q_weight_layer_name, old_q_weight_layer_name, dim=0)
                split_and_add_params(k_weight_layer_name, old_k_weight_layer_name, dim=0)
                split_and_add_params(v_weight_layer_name, old_v_weight_layer_name, dim=0)
            dense_weight_layer_name = f"layers.{layer_idx}.self_attention.dense.weight"
            old_dense_weight_layer_name = f"layers.{pp_layer_idx}.self_attention.dense.weight"
            split_and_add_params(dense_weight_layer_name, old_dense_weight_layer_name, dim=1)
            post_layernorm_name = f"layers.{layer_idx}.post_attention_layernorm.weight"
            old_post_layernorm_name = f"layers.{pp_layer_idx}.post_attention_layernorm.weight"
            for encoder in new_tp_encoders:
                encoder[post_layernorm_name] = full_encoder[old_post_layernorm_name].clone()
            # mlp_h_to_4h_name = f"layers.{layer_idx}.mlp.dense_h_to_4h.weight"
            # split_and_add_params(mlp_h_to_4h_name, dim=0)
            mlp_h_to_4h_name = f"layers.{layer_idx}.mlp.dense_h_to_4h.weight"
            old_mlp_h_to_4h_name = f"layers.{pp_layer_idx}.mlp.dense_h_to_4h.weight"
            mlp_h_to_4h_param = full_encoder[old_mlp_h_to_4h_name]
            mlp_h_to_4h_params = torch.chunk(mlp_h_to_4h_param, 2 * TP, dim=0)
            for tp_index in range(TP):
                encoder = new_tp_encoders[tp_index]
                encoder[mlp_h_to_4h_name] = torch.cat(
                        [mlp_h_to_4h_params[tp_index], mlp_h_to_4h_params[tp_index + TP]],
                        dim=0,
                    ).clone()
            mlp_4h_to_h_name = f"layers.{layer_idx}.mlp.dense_4h_to_h.weight"
            old_mlp_4h_to_h_name = f"layers.{pp_layer_idx}.mlp.dense_4h_to_h.weight"
            split_and_add_params(mlp_4h_to_h_name, old_mlp_4h_to_h_name, dim=1)
        if pp_rank == PP - 1:
            for encoder in new_tp_encoders:
                encoder['final_layernorm.weight'] = full_encoder['final_layernorm.weight'].clone()
        for tp in range(TP):
            if PP == 1:
                tp_dir_name = f"mp_rank_{tp:02d}"
            else:
                tp_dir_name = f"mp_rank_{tp:02d}_{pp_rank:03d}"
            current_language_model = OrderedDict()
            current_language_model['encoder'] = new_tp_encoders[tp]
            if pp_rank == 0:
                current_language_model['embedding'] = {
                    'word_embeddings': {
                        'weight': new_tp_word_embeddings[tp].clone()
                    }
                }
            if pp_rank == PP - 1:
                current_language_model['output_layer'] = {
                    'weight': new_tp_lm_heads[tp].clone()
                }
            saved_model = OrderedDict()
            for key in saved_info:
                saved_model[key] = copy.deepcopy(saved_info[key])
            saved_model['model'] = {
                'language_model': current_language_model,
            }
            os.makedirs(os.path.join(output_dir, tp_dir_name), exist_ok=True)
            print(f"Save tensor parallel {tp} pipeline parallel {pp_rank} to {os.path.join(output_dir, tp_dir_name)}")
            torch.save(saved_model, os.path.join(output_dir, tp_dir_name, 'model_rng.pt'))
            for dp in range(DP):
                if PP == 1:
                    dp_dir_name = f"mp_rank_{tp:02d}_{dp:03d}"
                else:
                    dp_dir_name = f"mp_rank_{tp:02d}_{pp_rank:03d}_{dp:03d}"
                optim_dir = os.path.join(output_dir, dp_dir_name)
                os.makedirs(optim_dir, exist_ok=True)
                torch.save(
                    dummy_optim_state_dict,
                    os.path.join(optim_dir, "optim.pt"),
                )
    if iter_name != 'release':
        iteration = int(iter_name.split("_")[-1])
    else:
        iteration = 'release'
    with open(os.path.join(args.target_model_dir, 'latest_checkpointed_iteration.txt'), 'w') as f:
        f.write(str(iteration))

