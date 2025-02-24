from typing import List
from utils.logger import Logger
from utils.mesh import SponjMesh

class SponjMeshGraph(Logger):
    def __init__(self, faces: List[int]):
        self.children = []
        self.faces = faces
        self.collapsed = False
    
    def merge(self, graph):
        self.faces += graph.faces 
        self.children += graph.children 

    def add_child(self, child):
        self.children.append(child)
    
    def collapse(self, k = 0):
        graph_c = SponjMeshGraph(self.faces[:])

        for child in self.children:
            if child.collapsed: continue

            child_c = child.collapse(k)
            if k + 2 - len(child_c.faces) - len(graph_c.faces) > 0:
                graph_c.merge(child_c)
            else:
                graph_c.add_child(child_c)
        self.collapsed = True
        return graph_c
    
    def get_face_matrix(self) -> List[List[int]]:
        face_matrix = [self.faces[:]]

        for child in self.children:
            face_matrix += child.get_face_matrix()
        return face_matrix

    def __str__(self):
        return f"Faces: {self.faces}, Children: {len(self.children)}"

def mesh_to_graph(mesh: SponjMesh):
    seen = {0}
    queue = [0]
    e_to_f = {}
    for i, face in enumerate(mesh.faces.tolist()):
        for j in range(len(face)):
            k = j + 1 if j < len(face) - 1 else 0
            v_1, v_2 = face[j], face[k]

            edge = (min(v_1, v_2), max(v_1, v_2))
            e_to_f[edge] = e_to_f.get(edge, set()) | {i}
     
    face_face_adj = [set() for _ in range(mesh.face_shape[0])]
    for e in e_to_f:
        for f in e_to_f[e]:
            face_face_adj[f] |= e_to_f[e]
   
    graphs = [SponjMeshGraph([i]) for i in range(mesh.face_shape[0])]

    while queue:
        n = len(queue)
        for _ in range(n):
            i = queue.pop(0)

            for j in face_face_adj[i]:
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
                    graphs[i].add_child(graphs[j])

    return graphs[0]
 
        
