## InfoCSE: Information-aggregated Contrastive Learning of Sentence Embeddings

This repository contains the code and pre-trained models for our EMNLP 2022 paper [InfoCSE: Information-aggregated Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2210.06432).

Thanks to SimCSE(Gao et al., EMNLP 2021)! Our work mainly based on [SimCSE repo](https://github.com/princeton-nlp/SimCSE).

## Quick Links

  - [Overview](#overview)
  - [Model List](#model-list)
  - [Use InfoCSE with Huggingface](#use-infocse-with-huggingface)
  - [Train InfoCSE](#train-infocse)
    - [Requirements](#requirements)
    - [Evaluation](#evaluation)
    - [Training](#training)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

Contrastive learning has been extensively studied in sentence embedding learning, which assumes that the embeddings of different views of the same sentence are closer. The constraint brought by this assumption is weak, and a good sentence representation should also be able to reconstruct the original sentence fragments. Therefore, this paper proposes an information-aggregated contrastive learning framework for learning unsupervised sentence embeddings, termed InfoCSE.
InfoCSE forces the representation of CLS positions to aggregate denser sentence information by introducing an additional Masked language model task and a well-designed network.

![](InfoCSE.png)

## Model List

Our released models are listed as following. You can import these models by using [HuggingFace's Transformers](https://github.com/huggingface/transformers).
|              Model              | Avg. STS |
|:-------------------------------|:--------:|
| [InfoCSE-bert-base](https://huggingface.co/ffgcc/InfoCSE-bert-base) |   78.85 |
| [InfoCSE-bert-base-rtd](https://huggingface.co/caskcsg/InfoCSE-bert-base-rtd) | 79.39 |
| [InfoCSE-bert-large](https://huggingface.co/ffgcc/InfoCSE-bert-large) |   80.18  |

## Use infocse with Huggingface

Besides using our provided sentence embedding tool, you can also easily import our models with HuggingFace's `transformers`:
```python
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("ffgcc/InfoCSE-bert-base")
model = AutoModel.from_pretrained("ffgcc/InfoCSE-bert-base")

# Tokenize input texts
texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
```

If you encounter any problem when directly loading the models by HuggingFace's API, you can also download the models manually from the above table and use `model = AutoModel.from_pretrained({PATH TO THE DOWNLOAD MODEL})`.

## Train infocse

In the following section, we describe how to train a infocse model by using our code.

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.7.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.1` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.1
```


Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

### Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. For STS tasks, our evaluation takes the "all" setting, and report Spearman's correlation. 

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

Then come back to the root directory, you can evaluate any `transformers`-based pre-trained models using our evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path ffgcc/InfoCSE-bert-base \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
```
which is expected to output the results in a tabular format:
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 70.53 | 84.59 | 76.40 | 85.10 | 81.95 |    82.00     |      71.37      | 78.85 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

Arguments for the evaluation script are as follows,

* `--model_name_or_path`: The name or path of a `transformers`-based pre-trained checkpoint. You can directly use the models in the above table, e.g., `ffgcc/InfoCSE-bert-base`.
* `--pooler`: Pooling method. Now we support
    * `cls` (default): Use the representation of `[CLS]` token. A linear+activation layer is applied after the representation (it's in the standard BERT implementation). 
    * `cls_before_pooler`: Use the representation of `[CLS]` token without the extra linear+activation. 
    * `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
    * `avg_top2`: Average embeddings of the last two layers.
    * `avg_first_last`: Average embeddings of the first and last layers. If you use vanilla BERT or RoBERTa, this works the best.
* `--mode`: Evaluation mode
    * `test` (default): The default test mode. To faithfully reproduce our results, you should use this option.
    * `dev`: Report the development set results. Note that in STS tasks, only `STS-B` and `SICK-R` have development sets, so we only report their numbers. It also takes a fast mode for transfer tasks, so the running time is much shorter than the `test` mode (though numbers are slightly lower).
    * `fasttest`: It is the same as `test`, but with a fast mode so the running time is much shorter, but the reported numbers may be lower (only for transfer tasks).
* `--task_set`: What set of tasks to evaluate on (if set, it will override `--tasks`)
    * `sts` (default): Evaluate on STS tasks, including `STS 12~16`, `STS-B` and `SICK-R`. This is the most commonly-used set of tasks to evaluate the quality of sentence embeddings.
    * `transfer`: Evaluate on transfer tasks.
    * `full`: Evaluate on both STS and transfer tasks.
    * `na`: Manually set tasks by `--tasks`.
* `--tasks`: Specify which dataset(s) to evaluate on. Will be overridden if `--task_set` is not `na`. See the code for a full list of tasks.

### Training

**Data**

Following SimCSE, we sample 1 million sentences from English Wikipedia; You can run `data/download_wiki.sh` to download the two datasets.

**Training scripts**

We provide example training scripts for infocse. In `Infocse-bert-base.sh`, we provide a single-GPU (or CPU) example.


We explain the arguments in following:
* `--train_file`: Training file path. We support "txt" files (one line for one sentence) . You can use Wikipedia or you can use your own data with the same format.
* `--model_name_or_path`: For the base model, we init our model with [Luyu/condenser](https://huggingface.co/Luyu/condenser). For the large model, we first reproduce the condenser-bert-large since there is no official model. We release the model in [https://huggingface.co/ffgcc/condenser-bert-large-uncased](https://huggingface.co/ffgcc/condenser-bert-large-uncased).
* `--temp`: Temperature for the contrastive loss.
* `--pooler_type`: Pooling method. It's the same as the `--pooler_type` in the [evaluation part](#evaluation).
* `--hard_negative_weight`: If using hard negatives (i.e., there are 3 columns in the training file), this is the logarithm of the weight. For example, if the weight is 1, then this argument should be set as 0 (default value).
* `--do_mlm`: Whether to use the MLM auxiliary objective. If True:
  * `--mlm_weight`: Weight for the MLM objective.
  * `--mlm_probability`: Masking rate for the MLM objective.

All the other arguments are standard Huggingface's `transformers` training arguments. Some of the often-used arguments are: `--output_dir`, `--learning_rate`, `--per_device_train_batch_size`. In our example scripts, we also set to evaluate the model on the STS-B development set (need to download the dataset following the [evaluation](#evaluation) section) and save the best checkpoint.

For results in the paper, we use Nvidia 3090 GPUs with CUDA 11. Using different types of devices or different versions of CUDA/other softwares may lead to slightly different performance.


## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Wuxing (`wuxing@iie.ac.cn`) and Chaochen (`gaochaochen@iie.ac.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!
