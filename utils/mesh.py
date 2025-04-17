import os
import base64
import trimesh 
import imageio
import pymeshlab
import numpy as np 
from utils import BASE_DIR
from utils.logger import Logger
import matplotlib.pyplot as plt
from utils.img import fig_to_img
from matplotlib.collections import PolyCollection

class SponjMesh(Logger):
    def __init__(self, obj_path: str = None, glb_path: str = None, faces=None, vertices=None, extract_colors=True, **kwargs):
        super().__init__()
        self.log_path = f"{BASE_DIR}/logs/sponj_mesh.log"

        self.obj_path = obj_path
        self.glb_path = glb_path
        self.ms = pymeshlab.MeshSet()
        if glb_path is not None and extract_colors:
            mesh = self.extract_colors_from_glb(glb_path, obj_path)
            self.ms.add_mesh(mesh)
        elif obj_path is not None:
            self.ms.load_new_mesh(obj_path)
            
        elif faces and vertices: 
            self.log(f"[SponjMesh] >> Creating mesh from faces, vertices and kwargs: {kwargs.keys()}")
            self.ms.add_mesh(pymeshlab.Mesh(face_matrix=faces, vertex_matrix=vertices, **kwargs))
        
        if self.ms.current_mesh().has_face_color():
            self.ms.compute_color_transfer_face_to_vertex()
            
        self.update()
        self.gif = None 
    
    def extract_colors_from_glb(self, glb_path, obj_path) -> pymeshlab.Mesh:
        glb = trimesh.load(glb_path, force='mesh')
        colored_glb = glb.visual.to_color()
        vertex_colors = colored_glb.vertex_colors
        face_colors = trimesh.visual.color.vertex_to_face_color(vertex_colors, glb.faces) / 255

        tmp_ms = pymeshlab.MeshSet()
        tmp_ms.load_new_mesh(obj_path)

        faces = tmp_ms.current_mesh().face_matrix()
        vertices = tmp_ms.current_mesh().vertex_matrix()

        tmp_ms.clear() 
        del tmp_ms
        if face_colors.shape[0] == faces.shape[0]:
            return pymeshlab.Mesh(
                face_matrix=faces, 
                vertex_matrix=vertices, 
                f_color_matrix=face_colors,
                v_color_matrix=vertex_colors,
            )
        else:
            return pymeshlab.Mesh(
                face_matrix=faces, 
                vertex_matrix=vertices, 
                v_color_matrix=vertex_colors,
            )

    def remove_empty(self, save_path=None, **kwargs):
        self.call([
            self.ms.meshing_remove_null_faces,
        ], **kwargs)

        self.save(-1, save_path)

    def remove_duplicates(self, save_path=None, **kwargs):
        self.call([
            self.ms.meshing_remove_duplicate_faces,
            self.ms.meshing_remove_duplicate_vertices
        ], **kwargs)

        self.save(-1, save_path)

    def repair(self, save_path=None, **kwargs):
        self.call([
            self.ms.meshing_repair_non_manifold_edges,
            self.ms.meshing_repair_non_manifold_vertices
        ], **kwargs)

        self.save(-1, save_path)

    def get_largest_cc(self, save_path=None):
        self.ms.generate_splitting_by_connected_components()
        n = len(self.ms)

        k, num_faces = 0, 0
        for i in range(1, n):
            if self.ms[i].face_matrix().shape[0] > num_faces:
                k = i
                num_faces = self.ms[i].face_matrix().shape[0]
        
        self.save(k, save_path)
        self.ms.set_current_mesh(k)
        return self.ms[k]

    def fill_holes(self, save_path=None, **kwargs):
        self.repair()
        self.call([
            self.ms.meshing_close_holes,
        ], **kwargs)

        self.save(-1, save_path)

    def remesh(self, save_path=None, **kwargs):
        self.call([
            self.ms.meshing_isotropic_explicit_remeshing,
        ], **kwargs)

        self.save(-1, save_path)

    def marching_cubes(self, save_path=None, **kwargs):
        self.repair()
        self.call([
            self.ms.generate_marching_cubes_apss,
        ], **kwargs)

        self.save(-1, save_path)

    def edge_collapse(self, save_path=None, **kwargs):
        self.call([
            self.ms.meshing_decimation_quadric_edge_collapse,
        ], **kwargs)

        self.face_shape = self.ms.current_mesh().face_matrix().shape

        self.save(-1, save_path)

    def smoothen(self, save_path=None, **kwargs):
        self.call([
            # self.ms.apply_coord_hc_laplacian_smoothing, 
            self.ms.apply_coord_depth_smoothing, 
            self.ms.apply_coord_laplacian_smoothing_surface_preserving
        ], **kwargs)
    
        self.save(-1, save_path)

    def decimate(self, save_path=None, targetfacenum=20000):
        for _ in range(10):
            self.edge_collapse(targetfacenum=targetfacenum)

            if self.face_shape[0] <= targetfacenum and save_path: 
                self.save(-1, save_path)
                break
        if self.face_shape[0] > targetfacenum:
            print("[ERROR] Could not reach target face number")
        
        self.update()
    
    def call(self, funcs, **kwargs):
        for func in funcs:
            # func_arg_names = inspect.getargspec(func)

            # func_args = {}
            # for arg_name, arg in kwargs.items():
            #     if arg_name in func_arg_names:
            #         func_args[arg_name] = arg
            # print(f"Calling {func.__name__}({func_args})")
            func(**kwargs)
        self.update()
    
    def render(self, angle_x=20, angle_y=60, mesh_path=None, save_path=None, save_kwargs={}):
        if mesh_path is not None:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(mesh_path)

            F = ms.current_mesh().face_matrix()
            V = ms.current_mesh().vertex_matrix()

            if ms.current_mesh().has_vertex_color():
                ms.apply_color_brightness_contrast_gamma_per_vertex(gamma=2.2)
                ms.compute_color_transfer_vertex_to_face()
                C = ms.current_mesh().face_color_matrix()
            else:
                C = np.ones((F.shape[0], 3))  # White color (R=1, G=1, B=1)
        else:
            F = self.faces
            V = self.vertices
            # C = self.face_colors
            if self.face_colors is None:
                C = np.ones((F.shape[0], 3))  # White color (R=1, G=1, B=1)
            else:
                C = self.face_colors

        V = (V - (V.max(0) + V.min(0)) / 2) / max(V.max(0) - V.min(0)) # normalize vertices 
        MVP = perspective(25, 1, 1, 100) @ translate(0, 0, -3.5) @ rotate(angle_x, 'x') @ rotate(angle_y, 'y') # create model-view-projection matrix
        V = np.c_[V, np.ones(len(V))] @ MVP.T # transform vertices
        V /= V[:, 3].reshape(-1,1) # normalize
        V = V[F] # transform faces

        T =  V[:, :, :2] # get 2D vertices

        Z = -V[:, :, 2].mean(axis=1) # get 2D depth
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        I = np.argsort(Z)
        T, C = T[I,:], C[I,:]

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)
        ax.axis('off')

        collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="none")
        ax.add_collection(collection)

        img = fig_to_img(fig, save_kwargs)
        if save_path is not None:
            img.save(save_path)
        plt.close()
        return img

    def generate_gif(self, gif_path, mesh_path=None, frames=12, axis='y', save_kwargs={}):
        n = 360 / frames
        frames = [
            self.render(angle_x=20, angle_y=i*n, mesh_path=mesh_path, save_kwargs=save_kwargs)
            for i in range(frames)
        ]

        imageio.mimsave(gif_path, frames, loop=0)
        with open(gif_path, "rb") as gif:
            gif_bytes = gif.read()
        self.gif = base64.b64encode(gif_bytes).decode('utf-8')
    
    def update(self):
        self.ms.compute_normal_per_face()
        self.ms.compute_normal_per_vertex()

        currect_mesh = self.ms.current_mesh()
        self.faces = currect_mesh.face_matrix()
        self.face_normals = currect_mesh.face_normal_matrix()

        self.vertices = currect_mesh.vertex_matrix()
        self.vertex_normals = currect_mesh.vertex_normal_matrix()

        if currect_mesh.has_vertex_color():
            self.vertex_colors = currect_mesh.vertex_color_matrix()
        else:
            self.vertex_colors = np.ones((self.vertices.shape[0], 3))

        if currect_mesh.has_face_color():
            self.face_colors = currect_mesh.face_color_matrix()
        else:
            self.face_colors = np.ones((self.faces.shape[0], 3))

        self.face_shape = self.faces.shape
        self.vertex_shape = self.vertices.shape

    def save(self, k, save_path=None):
        if save_path is not None:
            if k != -1: self.ms.set_current_mesh(k)
            self.ms.save_current_mesh(save_path)
    
    def json(self):
        self.update()
        return {
            "gif": self.gif, # TODO: make gif optional
            "faces": self.faces.tolist(),
            "vertices": self.vertices.tolist(),
            "colors": self.vertex_colors.tolist(),
            "normals": self.vertex_normals.tolist(),

            "glb": open(self.glb_path, "rb") if self.glb_path else None,
        }

import numpy as np

def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M

def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def translate(x, y, z):
    return np.array([
        [1, 0, 0, x], [0, 1, 0, y],
        [0, 0, 1, z], [0, 0, 0, 1]
    ], dtype=float)

def rotate(theta, axis='y'):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)

    if axis == 'y':
        return np.array([
            [ c, 0, s, 0], [ 0, 1, 0, 0],
            [-s, 0, c, 0], [ 0, 0, 0, 1]
        ], dtype=float)

    if axis == 'x':
        return np.array([
            [1, 0,  0, 0], [0, c, -s, 0],
            [0, s,  c, 0], [0, 0,  0, 1]
        ], dtype=float)
