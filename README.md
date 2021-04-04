# PPFNet-tensorflow

Unoficial implementation of PFFNet for pointcloud class.

## Project architecture:

- src/
  - dataset.py : handles the data preprocessing, constructing the patches.
  - utils.py : Save point cloud to ply files, to open with cloud compare.
  - threeDMatch.py : reads the SUN3D dataset as used in 3DMatch, compute normals and position of the points.
  - shapenet.py : Reads the shapenet dataset, sample points on the 3D models.
  - loss.py : implementation of the N-tuple loss.
  - pointnet.py : pointnet of the PPFNet model. 

## Dataset used:

- [shapenet](https://shapenet.org/) 
- [SUN3D from 3DMatch](https://3dmatch.cs.princeton.edu/) 

## Frameworks:

- tensorflow 2
- jupyter
- trimesh (to load shapenet)
- plyfile (to export ply files)
- pandas (Reading shapenet split)
