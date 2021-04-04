import numpy as np
from plyfile import PlyData, PlyElement


def format_np_array(array):
    dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')]
    return np.array([tuple(p) for p in array], dtype=dtype)

def format_np_array_scalar(array, name="scalar"):
    dtype=[(name, 'f4')]
    return np.array([(s,) for s in array], dtype=dtype)

def export_ply(file_name, vertices, normals=None, scalar=None):
    points = PlyElement.describe(format_np_array(vertices), 'points')
    elements = [points]
    if normals is not None: 
        normals = PlyElement.describe(format_np_array(normals), 'normals')
        elements.append(normals)
    if scalar is not None: 
        scalar = PlyElement.describe(format_np_array_scalar(scalar), 'scalar')
        elements.append(scalar)
    
    PlyData(elements).write(file_name)