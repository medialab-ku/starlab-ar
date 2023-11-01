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

**Step 5**: Execute the model
```bash
python -m demo.demo_bodymocap --input_path webcam  --out_dir ./mocap_output  --single_person --no_display
```

In this step, you should install many modules that are currently missing to run model successfully because this virtual environment is currently empty.

To solve the problems, you need to find the modules through pip and install them one by one. The modules required for each model are listed below.

```
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
python -m pip install torch torchvision torchaudio opencv-python torchgeometry smplx loguru yacs timm flatten-dict pytorch-lightning scipy pose3d 
```