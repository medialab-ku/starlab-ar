# ShapeNet CRN

## About

Convert dataset from ShapeNet Core to Cascaded Refinement Network

- ShapeNet Core
  - [Website](https://shapenet.org)
- Cascaded Refinement Network for Point Cloud Completion (CVPR 2020)
  - [GitHub](https://github.com/xiaogangw/cascaded-point-completion)

## Requirements

### Python

- Open3D

```shell
pip install open3d
```

- h5py
  - Python interface to the HDF5 binary data format

```shell
pip install h5py
```

- trimesh
  - Auto-triangulation for quad mesh

```shell
pip install trimesh
```

### C++

```shell
sudo apt install freeglut3-dev libglm-dev
```

## Build

- Build virtual scan program by CMake

```shell
mkdir build && cd build
cmake ..
make
```

## Run

- `ShapeNetCore.v2`
  - Prepare ShapeNetCore dataset
- `vscan/list.py`
  - From ShapeNetCore dataset
  - Create list of model IDs
- `build/vscan`
  - From ShapeNetCore dataset and list of model IDs
  - Create virtual scan point clouds
- `main.py`
  - From ShapeNetCore dataset and virtual scan point clouds
  - Create CRN dataset files(`.h5`)
