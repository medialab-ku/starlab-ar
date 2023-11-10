import taichi as ti


@ti.data_oriented
class TaichiRenderer():
    # based on pinhole camera model
    def __init__(self, input_size, mesh, camera, bg=ti.math.vec3([0.0, 0.0, 0.0])):
        self.w = input_size
        self.h = input_size
        self.mesh = mesh
        self.camera = camera
        self.background = ti.math.vec3(bg[0], bg[1], bg[2])

        self.num_v = mesh.v.shape[0]
        self.num_f = mesh.num_f

        self.ndc_verts = ti.Vector.field(3, dtype=ti.f32, shape=mesh.num_v)
        self.ndc_verts.fill(0.0)
        self.frag_pos = ti.Vector.field(3, dtype=ti.f32, shape=mesh.num_v)
        self.frag_pos.fill(0.0)

        self.vbo = ti.Vector.field(3, dtype=ti.f32, shape=mesh.num_v)
        self.vbo.from_numpy(mesh.v.to_numpy())
        self.ebo = ti.Vector.field(3, dtype=ti.i32, shape=mesh.num_f)
        self.ebo.from_numpy(mesh.f.to_numpy())

        self.z_buffer = ti.field(dtype=ti.f32, shape=(w, h))
        self.z_buffer.fill(-1000000.0)
        self.pixel = ti.Vector.field(3, dtype=float, shape=(w, h))

        self.light_pos = ti.Vector([15.0, 15.0, 15.0])
        self.light_color = ti.Vector([0.7, 0.7, 0.7])
        self.initialize()

        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=mesh.num_v)
        self.normal.from_numpy(mesh.normal.to_numpy())

    @ti.kernel
    def initialize(self):
        for i, j in self.z_buffer:
            self.z_buffer[i, j] = 1.0
            self.pixel[i, j] = self.background

    @ti.kernel
    def vertex_shader(self):
        for i in self.mesh.v:
            mvp = self.camera.mat_proj @ self.camera.mat_view @ self.mesh.mat_model
            point = self.mesh.v[i]
            model_p = mvp @ ti.Vector([point[0], point[1], point[2], 1.0])
            model_p /= model_p[3]

            self.ndc_verts[i] = ti.Vector([model_p[0], model_p[1], model_p[2]])

            frag_pos = self.mesh.mat_model @ ti.Vector([point[0], point[1], point[2], 1.0])
            self.frag_pos[i] = ti.Vector([frag_pos[0], frag_pos[1], frag_pos[2]])

            self.normal[i] = self.mesh.inv_trans_model @ self.mesh.normal[i]


    @ti.func
    def tri_area(self, p1, p2, p3):
        # p1, p2, and p3 are 2D vectors
        l1 = ti.math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        l2 = ti.math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
        l3 = ti.math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
        p = 0.5 * (l1 + l2 + l3)
        area = ti.math.sqrt(p * (p - l1) * (p - l2) * (p - l3))
        return area


    @ti.kernel
    def rasterizer(self):
        # TODO: Improving how to find pixels that pass over edges
        for f in range(self.mesh.num_f):
            tri = ti.Vector([self.mesh.f[f, 0], self.mesh.f[f, 1], self.mesh.f[f, 2]])

            # get triangle vertices
            p1 = self.ndc_verts[tri[0]]
            p2 = self.ndc_verts[tri[1]]
            p3 = self.ndc_verts[tri[2]]

            # compute bounding box
            min_x = int(min(p1[0], p2[0], p3[0]) * self.w) + self.w // 2
            max_x = int(max(p1[0], p2[0], p3[0]) * self.w) + self.w // 2
            min_y = int(min(p1[1], p2[1], p3[1]) * self.h) + self.h // 2
            max_y = int(max(p1[1], p2[1], p3[1]) * self.h) + self.h // 2

            # clip bounding box
            min_x = max(min_x, 0)
            max_x = min(max_x, self.w)
            min_y = max(min_y, 0)
            max_y = min(max_y, self.h)

            # compute triangle area
            p1_2d = ti.Vector([p1[0], p1[1]])
            p2_2d = ti.Vector([p2[0], p2[1]])
            p3_2d = ti.Vector([p3[0], p3[1]])
            area = self.tri_area(p1_2d, p2_2d, p3_2d)

            # backface culling
            if area <= 0:
                continue

            # compute barycentric coordinates
            for x, y in ti.ndrange((min_x-1, max_x+1), (min_y-1, max_y+1)):
                p = ti.Vector([(x - self.w / 2.0) / self.w, (y - self.h / 2.0) / self.h])

                # check if pixel is inside triangle
                w0 = self.tri_area(p2_2d, p3_2d, p) / area
                w1 = self.tri_area(p3_2d, p1_2d, p) / area
                w2 = self.tri_area(p1_2d, p2_2d, p) / area

                # if inside triangle
                if w0 + w1 + w2 - 1.0 < 1e-3:
                    # compute depth
                    depth = w0 * p1[2] + w1 * p2[2] + w2 * p3[2]
                    if depth > self.z_buffer[x, y]:
                        self.z_buffer[x, y] = depth

                        # barycentric interpolation
                        vtTri = ti.Vector([self.mesh.vtIdx[f, 0], self.mesh.vtIdx[f, 1], self.mesh.vtIdx[f, 2]])
                        tex_pos = w0 * self.mesh.vt[vtTri[0]] + w1 * self.mesh.vt[vtTri[1]] + w2 * self.mesh.vt[vtTri[2]]
                        normal = (w0 * self.normal[tri[0]] + w1 * self.normal[tri[1]] + w2 * self.normal[tri[2]]).normalized()
                        fragment_pos = w0 * self.frag_pos[tri[0]] + w1 * self.frag_pos[tri[1]] + w2 * self.frag_pos[tri[2]]
                        light_dir = (self.light_pos - fragment_pos).normalized()
                        obj_color = self.mesh.get_tex_color(tex_pos) / 255.0
                        view_dir = (self.camera.eye - fragment_pos).normalized()
                        color = self.fragment_shader(normal, obj_color, self.light_color, light_dir, 0.8, view_dir, 0.5)
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


    def visualize(self):
        # draw mesh
        self.vertex_shader()
        self.rasterizer()
        # self.swap_buffers()


    def imwrite(self, filepath: ti.template()):
        img = self.pixel.to_numpy()
        ti.tools.imwrite(img, filepath)
        print("Image saved to", filepath)
