# ACARL

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