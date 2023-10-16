# Fast Setup instructions

This quick instructions document contains 3 steps:

1. installing software
2. preparing data
3. running the script

This is useful if you need to ask someone to reproduce problems with `Megatron-LM`

## 1. Software

**NOTE that now we use docker to run Megatron-LM, so this section is not needed anymore.**

Please follow this exact order.


0. Create a new conda env if need be or activate an existing environment.

1. Install `pytorch`. Choose the desired version install instructions [here](https://pytorch.org/get-started/locally/), for pip it'd be:

```
pip install torch torchvision torchaudio
```

2. Install system-wide `cuda` if you don't have it already. [NVIDIA instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Of course ideally use [the premade packages for your distro](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation).
Use the same major version as pytorch's cuda build. To check use:

```
python -c 'import torch; print(f"pt={torch.__version__}, cuda={torch.version.cuda}")'
```

The minor versions don't actually have to match, but then you will need to hack `apex` installer to ignore minor version changes, see below.

3. Install `apex`

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
cd -
```

If the pytorch and system-wide cuda minor versions mismatch, it's not a problem, you just need to hack `apex`'s build to bypass the check by applying this patch first and then build it.
```
diff --git a/setup.py b/setup.py
index d76e998..f224dae 100644
--- a/setup.py
+++ b/setup.py
@@ -31,6 +31,8 @@ def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
     print(raw_output + "from " + cuda_dir + "/bin\n")

     if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
+        # allow minor diffs
+        if bare_metal_minor != torch_binary_minor: return
         raise RuntimeError(
             "Cuda extensions are being compiled with a version of Cuda that does "
             "not match the version used to compile Pytorch binaries.  "
```


4. Checkout and prepare `Megatron-LM` and install its requirements

```
git clone git@codeup.aliyun.com:deeplang/LLM/Megatron-LM.git
cd Megatron-LM
pip install -r requirements.txt
```




## 2. Data

Will work under the `Megatron-LM` clone

```
cd Megatron-LM
```



Prepare data for preprocessing
```
mkdir -p data
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O data/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O data/gpt2-merges.txt
python -c 'from datasets import load_dataset; ds = load_dataset("stas/oscar-en-10k", split="train", keep_in_memory=False); ds.to_json(f"data/oscar-en-10k.jsonl", orient="records", lines=True, force_ascii=False)'
```

Pre-process a small dataset to be used for training

```
python tools/preprocess_data.py \
    --input data/oscar-en-10k.jsonl \
    --output-prefix data/meg-gpt2-oscar-en-10k \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file data/gpt2-merges.txt \
    --vocab-file data/gpt2-vocab.json \
    --append-eod \
    --workers 4
```

now you have data/meg-gpt2-oscar-en-10k, vocab and merges files to pass as arguments to training, the next section shows how to use them.

Note that Megatron wants `data/meg-gpt2-oscar-en-10k_text_document` prefix later in `--data-path`

## 3. Train

`quick_run.sh` is a tiny model training script configured over 8 gpus to train on the data we prepared in step 2. Please change the settings in it and run it by:

```
sh quick_run.sh
```

Remember to wipe out `$CHECKPOINT_PATH`, if you change the model shape and there is a checkpoint with the old shapes saved already.