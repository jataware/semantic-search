# Fine Tuning GPT2-XL
## Overview
The goal of fine tuning GPT2-XL is to create a state of the art text generation large language model with knowledge and context specific to a custom or proprietary corpus of documents. As a result of the fine-tuning procedure, the original model weights are updated to better specialize on the domain specific data and the task of interest.

This document outlines a process that Jataware has used to fine tune this model on a corpus of 21,353 documents. The fine tuning process largely relies on this git repo https://github.com/Xirider/finetune-gpt2xl, which is targeted specifically at GPT2-XL, and GPT-NEO. For other architectures of language models, a fair amount of engineering work would be required to create a custom tuning process for that model and its use case (e.g. classification, summarization, question answering, etc).

Rigorously quantifying how much benefit fine tuning provides has proven difficult, but anecdotally we were able to confirm that through fine tuning, the model successfully learned new concepts and how to apply them: GPT2 was trained in February of 2019, before the start of the COVID-19 pandemic, so the base model had no knowledge of that term. When testing a use case for causality identification, the base model never used the term, whereas after fine tuning on a dataset of scientific papers, it was able to successfully indicate that one of the main causes of death in the United States was COVID-19.

The general approach for fine tuning is:
1. vm setup
1. clone/prereqs
1. insert your corpus
1. run training
1. extract trained model for later usage

## VM Setup
For our fine tuning experiment, we used an AWS EC2 p3.2xlarge instance (i.e. 1 tesla V100 GPU with 61BG of VRAM). We started with a fresh ubuntu 22.04 image, but alternatively, you could use one of the premade deep-learning amazon machine images. If you use a blank ubuntu image, you will need to manually install the nvidia drivers, e.g.:
    
```bash
$ sudo apt update
$ sudo apt install nvidia-driver-525-server
```

you should then be able to check that the nvidia drivers are installed and running:

```bash
$ nvidia-smi
```

If it doesn't work, you may need to reboot the VM.

## Clone/Prereqs

I use anaconda for managing python environments (and some special packages), and pip for the majority of of packages.
1. Install anaconda
1. Create a virtual environment:

    ```bash
    $ conda create -n finetune python=3.10
    $ conda activate finetune
    ```

1. Clone the repo

    ```bash
    $ git clone git@github.com:Xirider/finetune-gpt2xl.git
    ```

1. Install the requirements

    ```bash
    $ pip install -r requirements.txt
    ```


1. check that nvcc cuda to matches (or possibly exceeds) version of cuda in pytorch:

    ```python
    import torch
    print(torch.version.cuda)
    ```

    should match with

    ```bash
    nvcc --version
    ```

    if not a match (or nvcc isn't installed):

    ```bash
    conda install -c nvidia cuda-toolkit
    ```

    or for a specific version (see: https://anaconda.org/nvidia/cuda-toolkit):
    
    ```bash
    conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
    ```

    **NOTE**: Do not install nvcc with apt, as this will likely break the systems nvidia drivers. After installing nvcc, run `nvidia-smi` to verify that the drivers are still working.
    
    **NOTE** Do not use the nvcc from the conda-forge channel, as it is too old

    restart may be required after installing nvcc


## Custom Training Corpus
For data, the training script expects training examples to be split between the `train.csv` and `validation.csv` files. The repo provides a `text2csv.py` script which will convert `train.txt` and `validation.txt` to the correct format, where all of the corpus text is concatenated into the single files. 


The format of the `.csv` files is as follows:

```csv
text
<text of first example>
<text of second example>
...
<text of last example>
```

By default, the `text2csv.py` script stuffs the entire corpus into a single line, so if you want to ensure that the corpus data maintains specific break points, it is worth having your own process for generating the `train.csv` and `validation.csv` files.

Important to be aware of is most corpus data will contain newlines, so they need to be properly escaped. Python's `csv.writer` class is great for automatically handling escaping newlines and other special characters. E.g.:

```python
import csv

train_data = ['example 1', 'example 2', ...]

with open('train.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text'])
    writer.writerows([[d] for d in train_data])
```


## Training
Training is performed via the command from the repo basically verbatim:

```bash
deepspeed --num_gpus=1 run_clm.py \
--deepspeed ds_config.json \
--model_name_or_path gpt2-xl \
--train_file train.csv \
--validation_file validation.csv \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--eval_steps 200 \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 8
```

The only thing that might be changed is `num_train_epochs`, `output_dir`, and maybe `num_gpus`. I haven't tested multi-GPU training, but I presume that it is just a matter of changing the `--num_gpus` flag to the number of GPUs available.

## Extracting/Using the Trained Model
Once the model is trained, the model weights are stored in the folder listed in the `--output_dir` flag (e.g. `finetuned` in the example above). Copy the folder to wherever you want to use it. 

**NOTE** that the folder may be very large ~25GB per checkpoint, but you should be safe to delete any of the checkpoint folders, as the model is stored in the top level `pytorch_model.bin` file.

To use the model, you can load it in the same way you might load a transformers pretrained model from huggingface, just replacing the model name with the path to the folder containing the model:

```python
from transformers import pipeline

model_dir = 'path/to/finetuned'

generator = pipeline('text-generation', model=model_dir)
# use like a normal pretrained model
```


## Notes
- Each checkpoint for training is 25GB, so make sure there’s enough space on the VM. Depending on how much training is to be done, you may need to set up a process for pruning the checkpoints
- Training times:
    - training 1 epoch of the Shakespeare example in the repo should take about 15 minutes
    - training 1 epoch of a research paper dataset with text from ~21,000 papers, then split into ~930,000 paragraphs, takes about 10 hours.
    - the estimate for time remaining is generally not very accurate. A decent (probably conservative) adjustment is to multiply it by 2.


