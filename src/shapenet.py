import glob
import os

import numpy as np
import pandas as pd 
import trimesh

import src

def get_shapenet_data(split="train"):
    paths = glob.glob(src.SHAPENET_MODELS)
    df = pd.read_csv(src.SHAPENET_SPLIT)
    df = df.set_index("modelId")
    res = []
    for path in paths:
        name = path.split(os.sep)[-3]
        if name in df.index and df.loc[name, "split"] == split:
            res.append(path)
    return res

def get_vertices(path):
    fuze_trimesh = trimesh.load(path)
    points = []
    if type(fuze_trimesh) == trimesh.scene.scene.Scene:
        for key in fuze_trimesh.geometry:
            points.append(fuze_trimesh.geometry[key].vertices)
    else:
        assert type(fuze_trimesh) == trimesh.base.Trimesh
        points.append(fuze_trimesh.vertices)
    return np.concatenate(points)

def get_area(triangles):
    v1 = triangles[:,1] - triangles[:,0]
    v2 = triangles[:,2] - triangles[:,1]
    area = np.linalg.norm(np.cross(v1,v2), axis=1)
    return area    

def get_normals(triangles):
    v1 = triangles[:,1] - triangles[:,0]
    v2 = triangles[:,2] - triangles[:,1]
    normals = np.cross(v1,v2)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals 

def sample_on_triangles(triangles):
    coord = np.random.uniform(size=(len(triangles), 2, 1))
    coord_mirror = 1 - coord
    invalid = coord.sum(axis=1).sum(axis=1) > 1
    coord[invalid] = coord_mirror[invalid]
    u0 = triangles[:,1] - triangles[:,0]
    u1 = triangles[:,2] - triangles[:,0]
    points = triangles[:,0] + u0*coord[:,0] + u1*coord[:,1]
    return points

def sample_points(path, nb=50000):
    fuze_trimesh = trimesh.load(path)
    triangles = []
    if type(fuze_trimesh) == trimesh.scene.scene.Scene:
        for key in fuze_trimesh.geometry:
            triangles.append(fuze_trimesh.geometry[key].triangles)
    else:
        assert type(fuze_trimesh) == trimesh.base.Trimesh
        triangles.append(fuze_trimesh.triangles)
    triangles = np.concatenate(triangles)
    area = get_area(triangles)
    proba = area / np.sum(area)
    samples = np.random.choice(len(proba), size=nb, p=proba)
    basis = np.take(triangles, samples, axis=0)
    points = sample_on_triangles(basis)
    
    normals = get_normals(basis)
    return points, normals