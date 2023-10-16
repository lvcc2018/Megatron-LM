import wandb
import re
import sys
import os

os.environ["WANDB_API_KEY"] = "352250992d7cbe3b2ea1ce625f424ee96d0f5737"
os.environ["WANDB_PROJECT"] = "Pretrain"
os.environ["WANDB_ENTITY"] = "deeplang-ai"
os.environ["WANDB_NAME"] = "DLM_70b_WS64_TP8_PP4_MAXLR3e-5_DROP0.1_pretrain_exp"
os.environ["WANDB_NOTES"] = "DLM_70b_WS64_TP8_PP4_MAXLR3e-5_DROP0.1_pretrain_exp log to wandb"
filepath = "/mnt/data/lvchuancheng/logs/DLM-3-70B/log_exp/DLM_70b_WS64_TP8_PP4_MAXLR3e-5_DROP0.1_pretrain_exp_63.log"

if __name__ == "__main__":
    wandb.init()
    with open(filepath, "r") as f:
        for linenum, line in enumerate(f):
            if line.startswith(" iteration"):
                # train log
                line = line.strip()
                metrics = line.split("|")
                metrics = [m.strip() for m in metrics]
                iteration = re.match(r"iteration(\s+)(\d+)/(.+)", metrics[0]).group(2)
                metric_dict = {}
                for metric in metrics[1:]:
                    try:
                        metric_name, metric_value = metric.split(":")
                    except ValueError as e:
                        continue
                    metric_value = metric_value.strip()
                    metric_dict[metric_name] = metric_value
                real_metric_dict = {}
                split_consume = False
                for key in metric_dict:
                    if "consumed samples of" in key:
                        split_consume = True
                        break
                if split_consume:
                    for key in metric_dict:
                        if "consumed samples of" in key:
                            split_name = key.split("consumed samples of")[1].strip()
                            metric_value = metric_dict[key]
                            real_metric_dict[f"consumed_samples/{split_name}"] = int(metric_value)
                else:
                    metric_value = metric_dict["consumed samples"]
                    real_metric_dict["consumed_samples"] = int(metric_value)
                if "learning rate" in metric_dict:
                    lr = metric_dict.pop("learning rate")
                    real_metric_dict["learning-rate"] = float(lr)
                if "grad norm" in metric_dict:
                    grad_norm = metric_dict.pop("grad norm")
                    real_metric_dict["grad-norm"] = float(grad_norm)
                if "loss scale" in metric_dict:
                    loss_scale = metric_dict.pop("loss scale")
                    real_metric_dict["loss-scale"] = float(loss_scale)
                if "elapsed time per iteration (ms)" in metric_dict:
                    elapsed_time = metric_dict.pop("elapsed time per iteration (ms)")
                    real_metric_dict["iteration-time"] = float(elapsed_time) / 1000
                for key in metric_dict:
                    if "lm loss" in key:
                        metric_value = metric_dict[key]
                        real_metric_dict[f"loss/{key}"] = float(metric_value)
                # log to wandb
                wandb.log(real_metric_dict, step=int(iteration))
            # TODO: add eval log
                            
                