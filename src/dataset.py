import numpy as np
from sklearn.neighbors import KDTree
import scipy.spatial
import tensorflow as tf

import src

def angle(u,v):
    cross = np.linalg.norm(np.cross(u, v), axis=-1)
    dot = np.sum(u*v, axis=-1)
    return np.arctan2(cross, dot)

def random_rotation(nb):
    rot = scipy.spatial.transform.Rotation.random(nb)
    return rot.as_matrix()

class PPFPatchExtractor:
    """
    From a point cloud:
    Sample the points that will be centers
    Using KD-tree find neighborhoods
    Then compute PPF on them.
    """
    def __init__(self, nb_patches=512, nb_points=256, tau=0.1, random_rot=True):
        self.nb_patches = nb_patches
        self.nb_points = nb_points
        self.tau = tau
        self.random_rot = random_rot
        
    def make_patches(self, cloud):
        points, normals = cloud
        centers_idx = np.random.choice(len(points), size=self.nb_patches, replace=False)
        centers = points[centers_idx]
        kd_tree = KDTree(points) 
        indexes = kd_tree.query(centers, k=self.nb_points, return_distance=False)
        patches = points[indexes]
        centers_normals = normals[centers_idx]
        patches_normals = normals[indexes]
        return centers, centers_normals, patches, patches_normals, centers_idx

    def compute_ppf(self, centers, centers_normals, patches, patches_normals):
        centers = centers[:,np.newaxis,:]
        centers_normals = centers_normals[:,np.newaxis,:]
        delta = patches - centers
        dist = np.linalg.norm(delta, axis=-1)
        angle_0 = angle(centers_normals, delta) 
        angle_1 = angle(patches_normals, delta) 
        angle_2 = angle(centers_normals, patches_normals) 
        if self.random_rot:
            rot = random_rotation(self.nb_patches)
            delta = np.einsum("ijk,ikl->ijl", delta, rot)
        ppf = np.stack((dist, angle_0, angle_1, angle_2), axis=-1)
        features = np.concatenate((delta, patches_normals, ppf), axis=-1)
        return features
    
    def compute_M(self, centers):
        diff = centers[:,np.newaxis] - centers[np.newaxis,:]
        dist = np.sum(diff*diff, axis=-1)
        M = dist < self.tau*self.tau
        #print(np.sum(np.sqrt(dist)*(1-M)) / np.sum(1-M)) # estimateing theta
        return M
    
    def make_example(self, cloud, return_centers_idx=False):
        """
        Transform point cloud with normals into PPF.
        """
        centers, centers_normals, patches, patches_normals, centers_idx =  self.make_patches(cloud)
        ppf = self.compute_ppf(centers, centers_normals, patches, patches_normals)
        M = self.compute_M(centers)
        if return_centers_idx:
            return ppf.astype(np.float32), M.astype(np.float32), centers_idx.astype(np.int32)
        return ppf.astype(np.float32), M.astype(np.float32)
    
