from collections import defaultdict
import glob
import os 
import pathlib 
import pickle 
                    
    
import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree
import tqdm
    
TEST_SCENES = """sun3d-home_at-home_at_scan1_2013_jan_1
sun3d-home_md-home_md_scan9_2012_sep_30
sun3d-hotel_uc-scan3
sun3d-hotel_umd-smaryland_hotel1
sun3d-hotel_umd-maryland_hotel3
sun3d-mit_76_studyroom-76-1studyroom2
sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika""".split("\n")
    
def get_sun3d_scenes(sun_path="data/rgbd-datasets", split=None):
    sun3d_scenes = glob.glob(os.path.join(sun_path, "sun3d-*/"))

    if split == "train":
        sun3d_scenes = [s for s in sun3d_scenes if not np.any([ts in s for ts in TEST_SCENES])]
    if split == "test":
        sun3d_scenes = [s for s in sun3d_scenes if np.any([ts in s for ts in TEST_SCENES])]
    return sun3d_scenes

def get_cloud(frame_path, return_color=False, sub_sample=False):
    camera_path = os.path.join(pathlib.Path(frame_path).parent.parent, "camera-intrinsics.txt")
    camera_int = np.genfromtxt(camera_path, dtype=np.float32, delimiter='\t')[:,:-1]
    frame_mat_path = frame_path.replace(".depth.png", ".pose.txt")
    frame_int = np.genfromtxt(frame_mat_path, dtype=np.float32, delimiter='\t')[:,:-1]
    depth = Image.open(frame_path)
    depth = np.array(depth)
    mask = np.where(depth > 0)
    if sub_sample:
        mask = (mask[0][::10], mask[1][::10])
    coord = np.stack((mask[1], mask[0], np.ones_like(mask[0]),), axis=1)
    
    camera_coord = coord@np.linalg.inv(camera_int).T 
    camera_coord = camera_coord * depth[mask][:,np.newaxis] 
    camera_coord = camera_coord/1000
    global_coord = np.concatenate([camera_coord, np.ones((len(camera_coord), 1))], axis=1)
    global_coord = global_coord@frame_int.T
    global_coord = global_coord[:,:3] / global_coord[:,3:] #coord 4 Should be 1.0
    if return_color:
        frame_color_path = frame_path.replace(".depth.png", ".color.png")
        colors = np.array(Image.open(frame_color_path))
        colors = colors[mask]
        return global_coord, colors
    
    return global_coord 

def scan_scene_voxels(scene_path, voxel_size=0.3, silent=True):
    scene_frames = glob.glob(os.path.join(scene_path, "seq-01", "*.depth.png"))
    voxels = defaultdict(list)
    for frame_path in tqdm.tqdm(scene_frames, disable=silent):
        cloud = get_cloud(frame_path, sub_sample=True)
        cloud /= voxel_size
        cloud = cloud.astype(np.int)
        frame_voxels, counts = np.unique(cloud, axis=0, return_counts=True)
        frame_voxels = frame_voxels[counts>200]
        for v in frame_voxels:
            voxels[tuple(v)].append(frame_path)
    return voxels

def sorted_tuple(a,b):
    if a<b:
        return a,b
    return b,a

def preprocess_scenes(scenes_path, output_path, max_nb_couples=10000, minimum_common_voxels=15):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    for scene_path in tqdm.tqdm(scenes_path):
        scene_name = scene_path.split(os.sep)[-2]
        out_path = os.path.join(output_path, f"{scene_name}.pkl")
        voxels = scan_scene_voxels(scene_path, voxel_size=0.3, silent=True)
        #voxels = {v:voxels[v] for v in voxels if len(voxels[v]) > 5}
        nb_common_voxels = defaultdict(lambda:defaultdict(int))
        frames = set()
        for v in voxels:
            for frame1 in voxels[v]:
                frames.add(frame1)
                for frame2 in voxels[v]:
                    nb_common_voxels[frame1][frame2] += 1 
        
        couples = set()
        frames_names = list(nb_common_voxels.keys())
        frames_idx = {name:i for i,name in enumerate(frames_names)}

        for f in frames_names:
            for f2 in nb_common_voxels[f]:
                if f!=f2 and nb_common_voxels[f][f2] > minimum_common_voxels:
                    couples.add(sorted_tuple(frames_idx[f],frames_idx[f2]))

        couples = list(couples)
        print(len(couples))
        if len(couples) > max_nb_couples:
            couples = [couples[i] for i in np.random.choice(len(couples), max_nb_couples)]
            
        with open(out_path, 'wb') as f:
            pickle.dump((couples, frames_names), f)

def gen_scene_couple(scene_path, preprocessed_dir):
    scene_name = scene_path.split(os.sep)[-2]
    preprocessed_path = os.path.join(preprocessed_dir, f"{scene_name}.pkl")    
    with open(preprocessed_path, 'rb') as f:
        couples, frames_names = pickle.load(f)
    np.random.shuffle(couples)
    for couple in couples:
        yield frames_names[couple[0]], frames_names[couple[1]]
        
def get_normals(cloud, idx=None, k=8):
    kdt = KDTree(cloud)
    neighbors = kdt.query(cloud[idx] if idx is not None else cloud, k=k, return_distance=False)
    neighbors_cloud = cloud[neighbors]
    neighbors_cloud -= neighbors_cloud.mean(axis=1, keepdims=True)
    #u @ (s[..., None] * vh) 
    _, _, vh = np.linalg.svd(neighbors_cloud)
    normals = vh[:,2]
    return normals