import numpy as np
import torch

from cmpl.shinv_model import ShapeInversion


"""
ShapeInversion Processor
"""

# create shape-inversion model object
model = ShapeInversion()

def run(pcd):

    # copy partial point cloud data
    pcd = pcd.astype(np.float32)
    partial = torch.from_numpy(pcd).cuda()

    # reset generator
    model.reset_G()

    # set target shape
    model.set_target(partial=partial)

    # initialize latent vector
    model.select_z()

    # apply shape-inversion
    return model.run()
