import taichi as ti
from .obj import readobj


@ti.data_oriented
class Mesh():
    def __init__(self, file_path, tex_path):
        obj = readobj(file_path)
        v = ti.field(ti.math.vec3, shape=obj['v'].shape[0])
        f = ti.field(ti.i32, shape=(obj['f'].shape[0], obj['f'].shape[1]))
        vtIdx = ti.field(ti.i32, shape=(obj['f'].shape[0], obj['f'].shape[1]))
        vt = ti.field(ti.math.vec2, shape=obj['vt'].shape[0])

        v.from_numpy(obj['v'])
        vt.from_numpy(obj['vt'])
        f.from_numpy(obj['f'][:, :, 0])
        vtIdx.from_numpy(obj['f'][:, :, 1])

        self.v = v
        self.f = f
        self.vt = vt
        self.vtIdx = vtIdx
        print('f', self.f[0, 0], self.f[0, 1], self.f[0, 2])
        print('vtIdx', self.vtIdx[0, 0], self.vtIdx[0, 1], self.vtIdx[0, 2])
        self.num_v = v.shape[0]
        self.num_f = f.shape[0]

        tex_np = ti.tools.imread(tex_path).astype(float)
        tex = ti.field(ti.math.vec3, shape=(tex_np.shape[0], tex_np.shape[1]))
        tex.from_numpy(tex_np)
        self.tex_img = tex

        self.color = ti.field(ti.math.vec3, shape=(self.num_v,))
        self.set_color()

        self.normal = ti.field(ti.math.vec3, shape=(self.num_v,))
        self.vert_count = ti.field(ti.i32, shape=(self.num_v,))
        self.set_normal()

        self.mat_model = self.set_model_mat(5.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        print('mat_model')
        print(self.mat_model)
        i_t_model_4x4 = self.mat_model.inverse().transpose()
        self.inv_trans_model = ti.Matrix([[i_t_model_4x4[0, 0], i_t_model_4x4[0, 1], i_t_model_4x4[0, 2]],
                                          [i_t_model_4x4[1, 0], i_t_model_4x4[1, 1], i_t_model_4x4[1, 2]],
                                          [i_t_model_4x4[2, 0], i_t_model_4x4[2, 1], i_t_model_4x4[2, 2]]])

    @ti.kernel
    def set_color(self):
        for i in range(self.num_v):
            # get color from texture
            u = self.vt[i][0]
            v = self.vt[i][1]
            tex_u = int(u * self.tex_img.shape[0])
            tex_v = int(v * self.tex_img.shape[1])
            self.color[i] = self.tex_img[tex_u, tex_v]

    @ti.kernel
    def set_normal(self):
        for f in range(self.num_f):
            tri = ti.Vector([self.f[f, 0], self.f[f, 1], self.f[f, 2]])
            v0 = self.v[tri[0]]
            v1 = self.v[tri[1]]
            v2 = self.v[tri[2]]
            normal = (v1 - v0).cross(v2 - v0).normalized()

            for i in range(3):
                self.normal[tri[i]] += normal
                self.vert_count[tri[i]] += 1

        for i in range(self.num_v):
            self.normal[i] /= self.vert_count[i]
            self.normal[i] = self.normal[i].normalized()


    @ti.kernel
    def set_model_mat(self, scale: ti.f32, rot: ti.math.vec3, trans: ti.math.vec3) -> ti.types.matrix(4, 4, ti.f32):
        S = ti.Matrix.identity(ti.f32, 4) * scale

        rad = ti.math.radians(rot)
        R = ti.cast(ti.math.rotation3d(rad[0], rad[1], rad[2]), ti.f32)

        T = ti.Matrix([[1.0, 0.0, 0.0, trans[0]],
                       [0.0, 1.0, 0.0, trans[1]],
                       [0.0, 0.0, 1.0, trans[2]],
                       [0.0, 0.0, 0.0, 1.0]])
        mat_model = S @ R @ T
        return mat_model



    @ti.func
    def get_tex_color(self, tex_pos):
        tex_u = tex_pos[0] * self.tex_img.shape[0]
        tex_v = tex_pos[1] * self.tex_img.shape[1]

        w_u = tex_u - ti.floor(tex_u)
        w_v = tex_v - ti.floor(tex_v)

        # bi-linear interpolation
        u = int(ti.floor(tex_u))
        v = int(ti.floor(tex_v))
        c00 = self.tex_img[u, v]
        c01 = self.tex_img[u, v + 1]
        c10 = self.tex_img[u + 1, v]
        c11 = self.tex_img[u + 1, v + 1]

        c0 = c00 * (1 - w_u) + c10 * w_u
        c1 = c01 * (1 - w_u) + c11 * w_u
        color = c0 * (1 - w_v) + c1 * w_v
        return color


