import numpy as np
import taichi as ti
import meshtaichi_patcher as patcher

@ti.data_oriented
class Mesh:

    def __init__(self,
                 model_path,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0,
                 tex_obj=None,
                 tex_path=None):

        self.mesh = patcher.load_mesh(model_path, relations=["FV", "EV", "VV", "VE"])
        self.mesh.verts.place({'m': ti.f32,
                               'x0': ti.math.vec3,
                               'x': ti.math.vec3,
                               'v': ti.math.vec3,
                               'f_ext': ti.math.vec3,
                               'y': ti.math.vec3,
                               'ld': ti.f32,
                               'x_k': ti.math.vec3,
                               'p': ti.math.vec3,
                               'nc': ti.uint32,
                               'deg': ti.uint32,
                               'dx': ti.math.vec3,
                               'g': ti.math.vec3,
                               'h': ti.f32,
                               'hc': ti.f32})



        self.mesh.verts.m.fill(1.0)
        self.mesh.verts.v.fill([0.0, 0.0, 0.0])
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.num_verts = len(self.mesh.verts)

        self.mesh.edges.place({'l0': ti.f32,
                               'ld': ti.f32,
                               'vid': ti.math.ivec2,
                               'x': ti.math.vec3,
                               'v': ti.math.vec3,
                               'hij': ti.math.mat3, # bounding sphere radius
                               'hinv': ti.math.mat2}) # bounding sphere radius

        # self.mesh.faces.place({'aabb_min': ti.math.vec3,
        #                        'aabb_max': ti.math.vec3})  # bounding sphere radius

        # self.setCenterToOrigin()
        self.face_indices = ti.field(dtype=ti.i32, shape=len(self.mesh.faces) * 3)
        self.edge_indices = ti.field(dtype=ti.i32, shape=len(self.mesh.edges) * 2)
        self.initFaceIndices()
        self.initEdgeIndices()

        self.trans = trans
        self.rot = rot
        self.scale = scale

        self.applyTransform()
        self.computeInitialLength()
        self.mesh.verts.x0.copy_from(self.mesh.verts.x)

        self.model_mat = self.computeModelMat()

        # Texture settings
        self.tex_obj_path = tex_obj
        self.tex_path = tex_path
        obj_f_shape = ti.Vector([1, 1], ti.i32)
        obj_vt_shape = ti.Vector([1, 1], ti.i32)
        if self.tex_obj_path is not None:
            obj = readobj(tex_obj)
            obj_f_shape = obj['f'].shape
            obj_vt_shape = obj['vt'].shape

        print(obj_f_shape, obj_vt_shape)
        self.vtIdx = ti.field(ti.i32, shape=(obj_f_shape[0], obj_f_shape[1]))
        self.vt = ti.field(ti.math.vec2, shape=obj_vt_shape[0])
        tex_np = np.zeros((1, 1, 3))
        if self.tex_path is not None:
            tex_np = ti.tools.imread(tex_path).astype(float)
        self.tex_img = ti.field(ti.math.vec3, shape=(tex_np.shape[0], tex_np.shape[1]))
        self.tex_img.from_numpy(tex_np)
        if self.tex_obj_path is not None:
            self.vtIdx.from_numpy(obj['f'][:, :, 1])
            self.vt.from_numpy(obj['vt'])
        else:
            self.vtIdx.fill(0)
            self.vt.fill(0.0)



    @ti.kernel
    def computeInitialLength(self):
        for e in self.mesh.edges:
            e.l0 = (e.verts[0].x - e.verts[1].x).norm()
            e.vid[0] = e.verts[0].id
            e.vid[1] = e.verts[1].id

    @ti.kernel
    def computeModelMat(self) -> ti.types.matrix(4, 4, ti.f32):
        s = ti.math.vec3(self.scale, self.scale, self.scale)
        S = ti.Matrix([[s[0], 0.0, 0.0, 0.0],
                        [0.0, s[1], 0.0, 0.0],
                        [0.0, 0.0, s[2], 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

        rad = ti.math.radians(self.rot)
        R = ti.cast(ti.math.rotation3d(rad[0], rad[1], rad[2]), ti.f32)

        T = ti.Matrix([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [self.trans[0], self.trans[1], self.trans[2], 1.0]])
        mat_model = S @ R @ T
        return mat_model
    


    @ti.kernel
    def initFaceIndices(self):
        for f in self.mesh.faces:
            self.face_indices[f.id * 3 + 0] = f.verts[0].id
            self.face_indices[f.id * 3 + 1] = f.verts[1].id
            self.face_indices[f.id * 3 + 2] = f.verts[2].id

    @ti.kernel
    def initEdgeIndices(self):
        for e in self.mesh.edges:
            self.edge_indices[e.id * 2 + 0] = e.verts[0].id
            self.edge_indices[e.id * 2 + 1] = e.verts[1].id

    @ti.kernel
    def setCenterToOrigin(self):

        center = ti.math.vec3(0, 0, 0)
        for v in self.mesh.verts:
            center += v.x

        center /= self.num_verts
        for v in self.mesh.verts:
            v.x -= center

    @ti.kernel
    def applyTransform(self):
        # self.setCenterToOrigin()

        for v in self.mesh.verts:
            v.x *= self.scale

        for v in self.mesh.verts:
            v_4d = ti.Vector([v.x[0], v.x[1], v.x[2], 1])
            rot_rad = ti.math.radians(self.rot)
            rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
            v.x = ti.Vector([rv[0], rv[1], rv[2]])

        for v in self.mesh.verts:
            v.x += self.trans

    @ti.func
    def get_tex_color(self, tex_pos):
        tex_u = tex_pos[0] * self.tex_img.shape[0]
        tex_v = tex_pos[1] * self.tex_img.shape[1]

        w_u = tex_u - ti.floor(tex_u)
        w_v = tex_v - ti.floor(tex_v)

        # bi-linear interpolation
        u = int(ti.floor(tex_u))
        v = int(ti.floor(tex_v))
        # print(u, v)

        c00 = self.tex_img[u, v]
        c01 = self.tex_img[u, v + 1]
        c10 = self.tex_img[u + 1, v]
        c11 = self.tex_img[u + 1, v + 1]

        c0 = c00 * (1 - w_u) + c10 * w_u
        c1 = c01 * (1 - w_u) + c11 * w_u
        color = c0 * (1 - w_v) + c1 * w_v
        color /= 255.0
        return color

@ti.kernel
def applyTransform(pos: ti.template(), scale: ti.f32, trans: ti.math.vec3, rot: ti.math.vec3):
    num_v = pos.shape[0]

    for i in range(num_v):
        pos[i] *= scale

    for i in range(num_v):
        v_4d = ti.Vector([pos[i][0], pos[i][1], pos[i][2], 1])
        rot_rad = ti.math.radians(rot)
        rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
        pos[i] = ti.Vector([rv[0], rv[1], rv[2]])

    for i in range(num_v):
        pos[i] += trans


@ti.kernel
def makeBox(minmax: ti.template(), pos: ti.template()):
    # minmax: [min: vec3, max: vec3], pos: 8x3
    for i in range(8):
        pos[i] = ti.Vector([minmax[i // 4][0], minmax[i // 2 % 2][1], minmax[i % 2][2]])


def _tri_append(faces, indices):
    if len(indices) == 3:
        faces.append(indices)
    elif len(indices) == 4:
        faces.append([indices[0], indices[1], indices[2]])
        faces.append([indices[2], indices[3], indices[0]])
    elif len(indices) > 4:
        for n in range(1, len(indices) - 1):
            faces.append([indices[0], indices[n], indices[n + 1]])
    else:
        assert False, len(indices)


def readobj(path, orient='xyz', scale=None, simple=False, usemtl=True, quadok=False):
    v = []
    vt = []
    vn = []
    faces = []
    usemtls = []
    mtllib = None

    if callable(getattr(path, 'read', None)):
        lines = path.readlines()
    else:
        with open(path, 'rb') as myfile:
            lines = myfile.readlines()

    # cache vertices
    for line in lines:
        line = line.strip()
        assert isinstance(line, bytes), f'BytesIO expected! (got {type(line)})'
        try:
            type, fields = line.split(maxsplit=1)
            fields = [float(_) for _ in fields.split()]
        except ValueError:
            continue

        if type == b'v':
            v.append(fields)
        elif type == b'vt':
            vt.append(fields)
        elif type == b'vn':
            vn.append(fields)

    # cache faces
    for line in lines:
        line = line.strip()
        try:
            type, fields = line.split(maxsplit=1)
            fields = fields.split()
        except ValueError:
            continue

        if type == b'mtllib':
            mtllib = fields[0]
            continue

        if type == b'usemtl':
            usemtls.append([len(faces), fields[0]])
            continue

        # line looks like 'f 5/1/1 1/2/1 4/3/1'
        # or 'f 314/380/494 382/400/494 388/550/494 506/551/494' for quads
        if type != b'f':
            continue

        # a field should look like '5/1/1'
        # for vertex/vertex UV coords/vertex Normal  (indexes number in the list)
        # the index in 'f 5/1/1 1/2/1 4/3/1' STARTS AT 1 !!!
        indices = [[int(_) - 1 if _ else 0 for _ in field.split(b'/')] for field in fields]

        if quadok:
            faces.append(indices)
        else:
            _tri_append(faces, indices)

    ret = {}
    ret['v'] = np.array([[0, 0, 0]], dtype=np.float32) if len(v) == 0 else np.array(v, dtype=np.float32)
    ret['vt'] = np.array([[0, 0]], dtype=np.float32) if len(vt) == 0 else np.array(vt, dtype=np.float32)
    ret['vn'] = np.array([[0, 0, 0]], dtype=np.float32) if len(vn) == 0 else np.array(vn, dtype=np.float32)
    ret['f'] = np.zeros((1, 3, 3), dtype=np.int32) if len(faces) == 0 else np.array(faces, dtype=np.int32)
    if usemtl:
        ret['usemtl'] = usemtls
        ret['mtllib'] = mtllib

    if orient is not None:
        objorient(ret, orient)
    if scale is not None:
        if scale == 'auto':
            objautoscale(ret)
        else:
            ret['v'] *= scale

    if simple:
        return ret['v'], ret['f'][:, :, 0]

    return ret


def writeobj(path, obj):
    if callable(getattr(path, 'write', None)):
        f = path
    else:
        f = open(path, 'w')
    with f:
        f.write('# OBJ file saved by tina.writeobj\n')
        f.write('# https://github.com/taichi-dev/taichi_three\n')
        for pos in obj['v']:
            f.write(f'v {" ".join(map(str, pos))}\n')
        if 'vt' in obj:
            for pos in obj['vt']:
                f.write(f'vt {" ".join(map(str, pos))}\n')
        if 'vn' in obj:
            for pos in obj['vn']:
                f.write(f'vn {" ".join(map(str, pos))}\n')
        if 'f' in obj:
            if len(obj['f'].shape) >= 3:
                for i, face in enumerate(obj['f']):
                    f.write(f'f {" ".join("/".join(map(str, f + 1)) for f in face)}\n')
            else:
                for i, face in enumerate(obj['f']):
                    f.write(f'f {" ".join("/".join([str(f + 1)] * 3) for f in face)}\n')


def objunpackmtls(obj):
    faces = obj['f']
    parts = {}
    ends = []
    for end, name in obj['usemtl']:
        ends.append(end)
    ends.append(len(faces))
    ends.pop(0)
    for end, (beg, name) in zip(ends, obj['usemtl']):
        if name in parts:
            parts[name] = np.concatenate([parts[name], faces[beg:end]], axis=0)
        else:
            parts[name] = faces[beg:end]
    for name in parts.keys():
        cur = {}
        cur['f'] = parts[name]
        cur['v'] = obj['v']
        cur['vn'] = obj['vn']
        cur['vt'] = obj['vt']
        parts[name] = cur
    return parts


def objmtlids(obj):
    faces = obj['f']
    mids = np.zeros(shape=len(faces), dtype=np.int32)
    ends = []
    for end, name in obj['usemtl']:
        ends.append(end)
    ends.append(len(faces))
    ends.pop(0)
    names = []
    for end, (beg, name) in zip(ends, obj['usemtl']):
        if name not in names:
            mids[beg:end] = len(names) + 1
            names.append(name)
        else:
            mids[beg:end] = names.index(name) + 1
    return mids


def objverts(obj):
    return obj['v'][obj['f'][:, :, 0]]


def objnorms(obj):
    return obj['vn'][obj['f'][:, :, 2]]


def objcoors(obj):
    return obj['vt'][obj['f'][:, :, 1]]


def objautoscale(obj):
    obj['v'] -= np.average(obj['v'], axis=0)
    obj['v'] /= np.max(np.abs(obj['v']))


def objorient(obj, orient):
    flip = False
    if orient.startswith('-'):
        flip = True
        orient = orient[1:]

    x, y, z = ['xyz'.index(o.lower()) for o in orient]
    fx, fy, fz = [o.isupper() for o in orient]

    if x != 0 or y != 1 or z != 2:
        obj['v'][:, (0, 1, 2)] = obj['v'][:, (x, y, z)]
        obj['vn'][:, (0, 1, 2)] = obj['vn'][:, (x, y, z)]

    for i, fi in enumerate([fx, fy, fz]):
        if fi:
            obj['v'][:, i] = -obj['v'][:, i]
            obj['vn'][:, i] = -obj['vn'][:, i]

    if flip:
        obj['f'][:, ::-1, :] = obj['f'][:, :, :]


def objmknorm(obj):
    fip = obj['f'][:, :, 0]
    fit = obj['f'][:, :, 1]
    p = obj['v'][fip]
    nrm = np.cross(p[:, 2] - p[:, 0], p[:, 1] - p[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    fin = np.arange(obj['f'].shape[0])[:, np.newaxis]
    fin = np.concatenate([fin for i in range(3)], axis=1)
    newf = np.array([fip, fit, fin]).swapaxes(1, 2).swapaxes(0, 2)
    obj['vn'] = nrm
    obj['f'] = newf