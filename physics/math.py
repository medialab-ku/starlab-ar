import taichi as ti

@ti.kernel
def create_batch_eyes(I: ti.template(), batch_size: ti.i32, mat_dim: ti.template()):
    # Taichi only supports ti.math.inverse for dimension < 5 matrices (23.11.02)
    for i in range(batch_size):
        I[i] = ti.math.eye(mat_dim) * (i + 1)

