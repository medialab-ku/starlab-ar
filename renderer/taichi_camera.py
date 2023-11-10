import taichi as ti

mat4 = ti.types.matrix(4, 4, ti.f32)

@ti.data_oriented
class Camera():
    def __init__(self, eye, zn, zf):
        # For view matrix
        self.eye = ti.Vector([eye[0], eye[1], eye[2]])
        self.up = ti.Vector([0.0, 1.0, 0.0])
        self.lookat = ti.Vector([0.0, 0.0, 0.0])

        # For projection matrix
        self.fov = 30.0
        self.aspect = 1.0
        self.zn = zn
        self.zf = zf

        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)
        self.mat_proj = self.init_mat_proj(self.fov, self.aspect, self.zn, self.zf)

        print('mat_view')
        print(self.mat_view)
        print('mat_proj')
        print(self.mat_proj)


    @ti.kernel
    def init_mat_view(self, eye: ti.math.vec3, lookat: ti.math.vec3, up: ti.math.vec3) -> ti.types.matrix(4, 4, ti.f32):
        zaxis = (lookat - eye).normalized()
        if eye[0] == lookat[0] and eye[1] == lookat[1] and eye[2] == lookat[2]:
            zaxis = ti.Vector([0.0, 0.0, 1.0])

        xaxis = (up.cross(zaxis)).normalized()
        yaxis = zaxis.cross(xaxis)

        # orientation = ti.Matrix([[xaxis[0], yaxis[0], zaxis[0], 0.0],
        #                          [xaxis[1], yaxis[1], zaxis[1], 0.0],
        #                          [xaxis[2], yaxis[2], zaxis[2], 0.0],
        #                          [0.0, 0.0, 0.0, 1.0]])
        # translation = ti.Matrix([[1.0, 0.0, 0.0, 0.0],
        #                          [0.0, 1.0, 0.0, 0.0],
        #                          [0.0, 0.0, 1.0, 0.0],
        #                          [-self.eye[0], -self.eye[1], -self.eye[2], 1.0]])
        # view_mat = orientation @ translation
        view_mat = ti.Matrix([[xaxis[0], xaxis[1], xaxis[2], -eye[0]],
                              [yaxis[0], yaxis[1], yaxis[2], -eye[1]],
                              [zaxis[0], zaxis[1], zaxis[2], -eye[2]],
                              [0.0, 0.0, 0.0, 1.0]])
        return view_mat

    @ti.kernel
    def init_mat_proj(self, fov: ti.f32, aspect: ti.f32, near: ti.f32, far: ti.f32) -> ti.types.matrix(4, 4, ti.f32):
        # zn: near plane, zf: far plane
        f = 1.0 / ti.math.tan(fov / 2.0)
        a = f / aspect
        b = (far + near) / (near - far)
        c = (2.0 * far * near) / (near - far)
        proj_mat = ti.Matrix([[f, 0.0, 0.0, 0.0],
                              [0.0, a, 0.0, 0.0],
                              [0.0, 0.0, b, c],
                              [0.0, 0.0, -1.0, 0.0]])
        return proj_mat


    def set_aspect(self, aspect):
        self.aspect = aspect
        self.mat_proj = self.init_mat_proj(self.zn, self.zf)

    def set_fov(self, fov):
        self.fov = fov
        self.mat_proj = self.init_mat_proj(self.zn, self.zf)

    def set_eye(self, eye):
        self.eye = ti.Vector([eye[0], eye[1], eye[2]])
        self.mat_view = self.init_mat_view()

    def set_lookat(self, lookat):
        self.lookat = ti.Vector([lookat[0], lookat[1], lookat[2]])
        self.mat_view = self.init_mat_view()

    def set_up(self, up):
        self.up = ti.Vector([up[0], up[1], up[2]])
        self.mat_view = self.init_mat_view()


    def camera_move_forward(self, step):
        self.eye += (self.lookat - self.eye).normalized() * step
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)

    def camera_move_backward(self, step):
        self.eye -= (self.lookat - self.eye).normalized() * step
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)

    def camera_move_left(self, step):
        self.eye -= (self.lookat - self.eye).cross(self.up).normalized() * step
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)

    def camera_move_right(self, step):
        self.eye += (self.lookat - self.eye).cross(self.up).normalized() * step
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)

    def camera_move_up(self, step):
        self.eye += self.up * step
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)

    def camera_move_down(self, step):
        self.eye -= self.up * step
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)

    def camera_rotate_left(self, angle):
        self.lookat = self.eye + (self.lookat - self.eye).normalized() * ti.math.cos(angle) + self.up.cross(self.lookat - self.eye).normalized() * ti.math.sin(angle)
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)

    def camera_rotate_right(self, angle):
        self.lookat = self.eye + (self.lookat - self.eye).normalized() * ti.math.cos(angle) - self.up.cross(self.lookat - self.eye).normalized() * ti.math.sin(angle)
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)

    def camera_rotate_up(self, angle):
        self.lookat = self.eye + (self.lookat - self.eye).normalized() * ti.math.cos(angle) + self.up.cross(self.lookat - self.eye).normalized() * ti.math.sin(angle)
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)

    def camera_rotate_down(self, angle):
        self.lookat = self.eye + (self.lookat - self.eye).normalized() * ti.math.cos(angle) - self.up.cross(self.lookat - self.eye).normalized() * ti.math.sin(angle)
        self.mat_view = self.init_mat_view(self.eye, self.lookat, self.up)


