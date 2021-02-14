# Shape Inversion

## Original Project

- [GitHub](https://github.com/junzhezhang/shape-inversion)

## Downloads

### CRN Datasets

- Download CRN dataset in `./crn_data/`
  - [Google Drive](https://drive.google.com/file/d/1YzQIR5LRmsePz6F-Q4EsbFqHSbslTh9x/view?usp=sharing)
- Download can dataset in `./crn_can/`
  - [Google Drive](https://drive.google.com/file/d/1hjzA4E1437iHOPnmI4jIVY2v--p-qAd1/view?usp=sharing)
- Download augmented can dataset in `./crn_can_aug/`
  - [Google Drive](https://drive.google.com/file/d/1uEcHcanYygRc4YSCX-IIpVZ6zU0ORXlg/view?usp=sharing)

### Pretrained Models

- Download pretrained models in `./pretrained_models/`
  - [Google Drive](https://drive.google.com/file/d/1dFNKsgwRQXFlidbBofVrSzvnXKi3KSae/view?usp=sharing)

## Requirements

### PyTorch

- Install PyTorch nightly version

```shell
pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
```

### Open3D

- Install Open3D

```shell
pip install open3d
```

### Others

```shell
pip install plyfile h5py Ninja matplotlib scipy tqdm
```

## Troubleshooting

### Missing tqdm

- Install tqdm

```shell
pip install tqdm
```

### CUDA out of memory

- Fix batch size to small
  - `diversity_search()` in `shape_inversion.py` line 226
  - `add_argument()` in `arguments.py` line 46

```shell
python trainer.py ... --batch_size 8
```

### Visualization

- Add `--visualize` option

```shell
python trainer.py ... --visualize
```

### Hard coded batch size

- Fix hard coded batch size
  - `shape_inversion.py` line 232
  - `calculate_fpd()` in `eval_treegan.py` line 93
  - `checkpoint_eval()` in `pretrain_treegan.py` line 174

### Hard coded number of samples

- Fix hard coded number of samples
  - `checkpoint_eval()` in `pretrain_treegan.py` line 174
  - `add_argument()` in `arguments.py` line 76

### Weird checkpoint path

- Fix save checkpoint path
  - `pretrain_treegan.py` line 285

### Unused and deprecated function

- Comment out `scipy.misc.imread()` import
  - `evaluation/FPD.py` line 8

### Wrong default argument

- Fix `--ckpt_load` default argument to `None`
  - `add_argument()` in `arguments.py` line 42

## Implementation

### Additional can label

- Add can label to CRN dataset
  - `data/CRN_dataset.py` line 29

### Registration process

- Add registraion process with Open3D
  - `registration.py`
  - `registration.run()` in `shape_inversion.py` line 153
  - `add_argument()` in `arguments.py` line 130
