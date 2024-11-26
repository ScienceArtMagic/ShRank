# ShRank (Shrink Rank)
### Automatic factorization library for pytorch

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Forked from [Greenformer](https://github.com/SamuelCahyawijaya/greenformer), the official implementation of the following paper:
- Winata, G. I., Cahyawijaya, S., Lin, Z., Liu, Z., & Fung, P. (2020, May). Lightweight and efficient end-to-end speech recognition using low-rank transformer. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6144-6148). IEEE.

### What is ShRank?
ShRank is a library to convert `Linear`, GPT-style "`Conv1D`" (really just transposed Linear), `Conv1d` (actual 1D convolution), `Conv2d`, `Conv3d` layers into its own variant which called `LED`/`CED`.
ShRank seeks over your PyTorch module, replace all `Linear`/GPT "`Conv1D`" layers into `LED` layers and all `Conv1d`, `Conv2d`, `Conv3d` layers into `CED` layers with the specified rank.

For the time being, ShRank is focused exclusively on using a faster, non-random implementation of SVD (`torch.linalg.svg` instead of `torch.svd_lowrank` used in Greenformer's SVD solver). Removed Greenformer's NMF and SNMF solvers (due to unmaintained dependency AFAICT), as well as the random (from scratch) solver.

### How to Install
```
pip install shrank
```

### Usage
##### BERT Model
```
from transformers import BertModel, BertConfig
from shrank import auto_fact

config = BertConfig.from_pretrained('bert-base-uncased', pretrained=False)
model = BertModel(config=config)

model = auto_fact(model, rank=100, deepcopy=False, fact_led_unit=False)
```

##### VGG Model
```
import torch
from torchvision import models
from shrank import auto_fact

model = models.vgg16()
model = auto_fact(model, rank=64, deepcopy=False, fact_led_unit=False)
```

### Why Use ShRank (WIP/TODO: update notebooks)
- Improve the speed of you model significantly, check our [Example Notebook](https://github.com/SamuelCahyawijaya/py_auto_fact/blob/main/examples/factorize_bert.ipynb)
- Maintain model performance with appropriate choice of rank, check our [ICASSP 2020 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053878)
- Easy to use and works on any kind of model!
