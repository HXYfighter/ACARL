# ACARL

This is the code repository for our paper published in Journal of Cheminformatics: [Activity cliff-aware reinforcement learning for _de novo_ drug design](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-01006-3).

Dependencies:

```bash
pytorch
numpy
pandas
tqdm
tensorboard
rdkit
openbabel
PyTDC
```

Run the `ACARL` algorithm (for example, design drug molecules against the 5HT1B target):

```bash
python codes/pretrain.py
python codes/RLtrain.py --target 5HT1B
```
