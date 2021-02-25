import numpy as np
from plyfile import PlyData, PlyElement


def format_np_array(array):
    dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')]
    return np.array([tuple(p) for p in array], dtype=dtype)

def export_ply(file_name, vertices, normals=None):
    points = PlyElement.describe(format_np_array(vertices), 'points')
    elements = [points]
    if normals is not None: 
        normals = PlyElement.describe(format_np_array(normals), 'normals')
        elements.append(elements)
    
    PlyData(elements).write(file_name)