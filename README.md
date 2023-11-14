# 3D_human_shape
3D real-time human shape estimation

## How to run (for example, 'frankmocap')

**Step 1**: Copy files that are large in size and have not been uploaded to GitHub

**Step 2**: Install virtualenv (https://pypi.org/project/virtualenv/)
```bash
sudo pip3 install virtualenv
```

**Step 3**: Create a virtual environment name frankmocap
```bash
python3 -m venv frankmocap
```


**Step 3-1(optional)**: Copy local environments into new virtual environment
```bash
virtualenv romp --system-site-packages
```

**Step 4**: Run virtual environment named frankmocap
```bash
source frankmocap/bin/activate
```

**Step 5**: Install modules required for each model
Each required module is listed below.

Also, you have to install other third-party libraries + download pretrained models and sample data.
    
```bash
sh scripts/install_frankmocap.sh
```

**Step 6**: Execute the model
```bash
python -m demo.demo_bodymocap --input_path webcam  --out_dir ./mocap_output  --single_person --no_display
```

If you have problem with the above command because of the relative path, try the following command.
```bash
export PYTHONPATH={path to 3D_human_shape}/frankmocap
python demo/demo_bodymocap.py --input_path webcam  --out_dir ./mocap_output  --single_person --no_display
```

```bash
Command line options:

- input_path
    - webcam
    - video
- out_dir
    - ./mocap_output
- single_person
- no_display
```

## Modules required for each model

### frankmocap
```bash
python -m pip install torch torchvision torchaudio opencv-python torchgeometry smplx loguru yacs timm flatten-dict pytorch-lightning scipy pose3d roma einops taichi meshtaichi_patcher
```