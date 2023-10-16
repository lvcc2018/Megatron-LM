import argparse
import os
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, help='Dir for target model', required=True)
    parser.add_argument("--tp", type=int, help="tensor parallel size", required=True)
    parser.add_argument("--pp", type=int, help="pipeline parallel size", required=True)
    parser.add_argument("--dp", type=int, help="pipeline parallel size", required=True)
    args = parser.parse_args()
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
    TP = args.tp
    PP = args.pp
    DP = args.dp
    for i in range(TP):
        for j in range(PP):
            for k in range(DP):
                if PP == 1:
                    checkpoint_dir = f"mp_rank_{i:02d}_{k:03d}"
                else:
                    checkpoint_dir = f"mp_rank_{i:02d}_{j:03d}_{k:03d}"
                checkpoint_dir = os.path.join(args.output_dir, checkpoint_dir)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    dummy_optim_state_dict,
                    os.path.join(checkpoint_dir, "optim.pt"),
                )
    