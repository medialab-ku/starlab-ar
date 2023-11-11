import taichi as ti
import numpy as np
import torch


@ti.data_oriented
class TaichiRenderer():
    # based on pinhole camera model
    def __init__(self, input_size, mesh, static_mesh, camera, bg=ti.math.vec3([1.0, 1.0, 1.0])):
        self.w = input_size
        self.h = input_size
        self.mesh = mesh
        self.static_mesh = static_mesh
        self.camera = camera
        self.background = ti.math.vec3(bg[0], bg[1], bg[2])

        self.mesh_num_v = mesh.mesh.verts.x.shape[0]
        self.mesh_num_f = mesh.face_indices.shape[0] // 3
        self.s_mesh_num_v = static_mesh.mesh.verts.x.shape[0]
        self.s_mesh_num_f = static_mesh.face_indices.shape[0] // 3

        self.num_v = self.mesh_num_v + self.s_mesh_num_v
        self.num_f = self.mesh_num_f + self.s_mesh_num_f

        self.mesh_v = ti.Vector.field(3, dtype=ti.f32, shape=self.mesh_num_v)
        self.mesh_v.from_numpy(mesh.mesh.verts.x.to_numpy())
        self.mesh_f = ti.Vector.field(3, dtype=ti.i32, shape=self.mesh_num_f)
        self.mesh_f.from_numpy(mesh.face_indices.to_numpy().reshape(-1, 3))

        self.s_mesh_v = ti.Vector.field(3, dtype=ti.f32, shape=self.s_mesh_num_v)
        self.s_mesh_v.from_numpy(static_mesh.mesh.verts.x.to_numpy())
        self.s_mesh_f = ti.Vector.field(3, dtype=ti.i32, shape=self.s_mesh_num_f)
        self.s_mesh_f.from_numpy(static_mesh.face_indices.to_numpy().reshape(-1, 3))

        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.num_v)
        self.v.from_numpy(np.concatenate((self.mesh_v.to_numpy(), self.s_mesh_v.to_numpy()), axis=0))
        self.f = ti.Vector.field(3, dtype=ti.i32, shape=self.num_f)
        self.f.from_numpy(np.concatenate((self.mesh_f.to_numpy(), self.s_mesh_f.to_numpy()+np.ones((self.s_mesh_num_f, 3))*(self.mesh_num_v)), axis=0))

        self.model_mat = mesh.model_mat
        self.inv_trans_mat = mesh.inv_trans_model
        self.s_model_mat = static_mesh.model_mat
        self.s_inv_trans_mat = static_mesh.inv_trans_model

        self.ndc_verts = ti.Vector.field(3, dtype=ti.f32, shape=self.num_v)
        self.ndc_verts.fill(0.0)
        self.frag_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.num_v)
        self.frag_pos.fill(0.0)

        self.z_buffer = ti.field(dtype=ti.f32, shape=(self.w, self.h))
        self.z_buffer.fill(-10.0)
        self.pixel = ti.Vector.field(3, dtype=float, shape=(self.w, self.h))

        self.light_pos = ti.Vector([15.0, 15.0, 15.0])
        self.light_color = ti.Vector([0.7, 0.7, 0.7])
        self.initialize_window()

        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=self.num_v)
        self.n_count = ti.field(dtype=ti.i32, shape=self.num_v)
        self.compute_normal()

        print(f'Renderer initialized: #v={self.num_v}, #f={self.num_f}')
        

    @ti.kernel
    def initialize_window(self):
        for i, j in self.z_buffer:
            self.z_buffer[i, j] = -10.0
            self.pixel[i, j] = self.background

    @ti.kernel
    def compute_normal(self):
        for i in range(self.num_f):
            tri = self.f[i]
            v0 = self.v[tri[0]]
            v1 = self.v[tri[1]]
            v2 = self.v[tri[2]]
            normal = (v1 - v0).cross(v2 - v0)
            self.normal[tri[0]] += normal
            self.normal[tri[1]] += normal
            self.normal[tri[2]] += normal
            self.n_count[tri[0]] += 1
            self.n_count[tri[1]] += 1
            self.n_count[tri[2]] += 1

        for i in range(self.num_v):
            self.normal[i] /= self.n_count[i]
            self.normal[i] = self.normal[i].normalized()

    @ti.kernel
    def vertex_shader(self, view_mat: ti.types.matrix(4, 4, ti.f32), proj_mat: ti.types.matrix(4, 4, ti.f32)):
        vp = view_mat @ proj_mat
        for i in range(self.num_v):
            model_mat = ti.Matrix.identity(ti.f32, 4)
            inv_trans_model = ti.Matrix.identity(ti.f32, 3)
            if i < self.mesh_num_v:
                model_mat = self.model_mat
                inv_trans_model = self.inv_trans_mat
            else:
                model_mat = self.s_model_mat
                inv_trans_model = self.s_inv_trans_mat
            mvp = model_mat @ vp
            point = self.v[i]
            model_mvp = ti.Vector([point[0], point[1], point[2], 1.0]) @ mvp
            model_mvp /= model_mvp[3]
            # print('model_p', self.v[i], model_p)

            self.ndc_verts[i] = ti.Vector([model_mvp[0], model_mvp[1], model_mvp[2]])

            frag_pos = ti.Vector([point[0], point[1], point[2], 1.0]) @ model_mat
            self.frag_pos[i] = ti.Vector([frag_pos[0], frag_pos[1], frag_pos[2]])


            self.normal[i] = self.normal[i] @ inv_trans_model

    @ti.func
    def view_port_transform(self, pos):
        vp_trans = ti.Matrix([[self.w / 2.0, 0.0, 0.0, 0.0],
                              [0.0, self.h / 2.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [self.w / 2.0, self.h / 2.0, 0.0, 1.0]])
        clip = ti.Vector([pos[0], pos[1], pos[2], 1.0]) @ vp_trans
        clip /= clip[3]
        return clip
    
    @ti.func
    def get_NDC_pos(self, pos):
        vp_trans_inv = ti.Matrix([[2.0 / self.w, 0.0, 0.0, 0.0],
                                    [0.0, 2.0 / self.h, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [-1.0, -1.0, 0.0, 1.0]])
        return pos @ vp_trans_inv

    @ti.func
    def tri_area(self, p1, p2, p3):
        # p1, p2, and p3 are 2D vectors
        area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - \
               (p3[0] - p1[0]) * (p2[1] - p1[1])
        return area
    
    @ti.func
    def dist_point_line(self, l1, l2, p):
        # l1, l2, and p are 2D vectors
        return (l2[1] - l1[1]) * p[0] - (l2[0] - l1[0]) * p[1] + l2[0] * l1[1] - l2[1] * l1[0]


    @ti.kernel
    def rasterizer(self):
        # TODO: Improving how to find pixels that pass over edges
        for f in range(self.num_f):
            color = ti.Vector([0.0, 0.0, 0.0])
            if f < self.mesh_num_f:
                color = ti.Vector([0.8, 0.53, 0.53])
            else:
                color = ti.Vector([0.65098039, 0.74117647, 0.85882353])
            tri = self.f[f]

            # get triangle vertices
            p1 = self.ndc_verts[tri[0]]
            p2 = self.ndc_verts[tri[1]]
            p3 = self.ndc_verts[tri[2]]

            c1 = self.view_port_transform(p1)
            c2 = self.view_port_transform(p2)
            c3 = self.view_port_transform(p3)

            # compute bounding box
            min_x = int(min(c1[0], c2[0], c3[0]))
            max_x = int(max(c1[0], c2[0], c3[0]))
            min_y = int(min(c1[1], c2[1], c3[1]))
            max_y = int(max(c1[1], c2[1], c3[1]))

            # clip bounding box
            min_x = max(min_x, 0)
            max_x = min(max_x, self.w)
            min_y = max(min_y, 0)
            max_y = min(max_y, self.h)

            # compute triangle area
            c1_2d = ti.Vector([c1[0], c1[1]])
            c2_2d = ti.Vector([c2[0], c2[1]])
            c3_2d = ti.Vector([c3[0], c3[1]])
            area = self.tri_area(c1_2d, c2_2d, c3_2d)
            # print(area)

            # backface culling
            if area < 0:
                continue

            # compute barycentric coordinates
            for x, y in ti.ndrange((min_x, max_x+1), (min_y, max_y)):
                p = ti.Vector([x+0.5, y+0.5])

                # check if pixel is inside triangle
                w0 = self.tri_area(c2_2d, c3_2d, p)
                w1 = self.tri_area(c3_2d, c1_2d, p)
                w2 = self.tri_area(c1_2d, c2_2d, p)

                # if inside triangle
                if w0 > 0.0 and w1 > 0.0 and w2 > 0.0:
                    # compute depth
                    depth = w0 * c1[2] + w1 * c2[2] + w2 * c3[2]
                    diff = depth - self.z_buffer[x, y]
                    # if ti.atomic_max(0.0, diff) > 0.0:
                    if diff > 0.0:
                        self.z_buffer[x, y] = depth

                        # barycentric interpolation
                        # vtTri = ti.Vector([self.mesh.vtIdx[f, 0], self.mesh.vtIdx[f, 1], self.mesh.vtIdx[f, 2]])
                        # tex_pos = w0 * self.mesh.vt[vtTri[0]] + w1 * self.mesh.vt[vtTri[1]] + w2 * self.mesh.vt[vtTri[2]]
                        # normal = (w0 * self.normal[tri[0]] + w1 * self.normal[tri[1]] + w2 * self.normal[tri[2]]).normalized()
                        # fragment_pos = w0 * self.frag_pos[tri[0]] + w1 * self.frag_pos[tri[1]] + w2 * self.frag_pos[tri[2]]
                        # light_dir = (self.light_pos - fragment_pos).normalized()
                        # obj_color = self.mesh.get_tex_color(tex_pos) / 255.0
                        # view_dir = (self.camera.eye - fragment_pos).normalized()
                        # color = self.fragment_shader(normal, obj_color, self.light_color, light_dir, 0.8, view_dir, 0.5)
                        self.pixel[x, y] = color
                

    @ti.func
    def fragment_shader(self, normal, obj_color, light_color, light_dir, ambient, view_dir, specular):
        # Phong shading
        # ambient
        ambient_color = ambient * obj_color

        # diffusion
        diff = max(0.0, light_dir.dot(normal))
        diffuse_color = diff * light_color

        # reflection
        reflect_dir = ti.math.reflect(-light_dir, normal)
        spec = ti.math.pow(max(0.0, reflect_dir.dot(view_dir)), 32)
        specular_color = specular * spec * light_color

        result = (ambient_color + diffuse_color + specular_color) * obj_color
        return result


    # @ti.kernel
    # def swap_buffers(self):
    #     for i, j in self.pixel:
    #         self.double_buffer[i, j] = self.pixel[i, j]


    def draw(self):
        # draw mesh
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix(self.w / self.h)
        self.initialize_window()
        self.vertex_shader(view_matrix, proj_matrix)
        self.rasterizer()
        # self.swap_buffers()

        
    def imwrite(self, filepath: ti.template()):
        img = self.pixel.to_numpy()
        ti.tools.imwrite(img, filepath)
        print("Image saved to", filepath)

    # def test(self):
    #     pos = ti.math.vec3([1.0, 1.0, 0.0])
    #     pos4d = ti.Vector([pos[0], pos[1], pos[2], 1.0])
    #     view = self.camera.get_view_matrix()
    #     view_mat = ti.Matrix([[view[0, 0], view[0, 1], view[0, 2], view[0, 3]],
    #                           [view[1, 0], view[1, 1], view[1, 2], view[1, 3]],
    #                           [view[2, 0], view[2, 1], view[2, 2], view[2, 3]],
    #                           [view[3, 0], view[3, 1], view[3, 2], view[3, 3]]])
    #     proj = self.camera.get_projection_matrix(self.w / self.h)
    #     proj = ti.Matrix([[proj[0, 0], proj[0, 1], proj[0, 2], proj[0, 3]],
    #                       [proj[1, 0], proj[1, 1], proj[1, 2], proj[1, 3]],
    #                       [proj[2, 0], proj[2, 1], proj[2, 2], proj[2, 3]],
    #                       [proj[3, 0], proj[3, 1], proj[3, 2], proj[3, 3]]])
    #     pos_model = pos4d @ self.model_mat
    #     pos_m_v = pos_model @ view_mat
    #     pos_m_v_p = pos_m_v @ proj
    #     print(pos_model)
    #     print(pos_m_v)
    #     print(pos_m_v_p)

    def set_camera(self, camera):
        self.camera = camera

    @ti.kernel
    def update_mesh_pos(self, pos: ti.template()):
        for i in range(self.mesh_num_v):
            self.mesh_v[i] = ti.Vector([pos[i][0], pos[i][1], pos[i][2]])
            self.v[i] = ti.Vector([pos[i][0], pos[i][1], pos[i][2]])

    @ti.kernel
    def update_static_mesh_pos(self, pos: ti.template()):
        for i in range(self.s_mesh_num_v):
            self.s_mesh_v[i] = ti.Vector([pos[i][0], pos[i][1], pos[i][2]])
            self.v[i+self.mesh_num_v] = ti.Vector([pos[i][0], pos[i][1], pos[i][2]])