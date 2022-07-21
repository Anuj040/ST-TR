import itertools

import numpy as np

"""
Function adapted from: https://github.com/kenziyuliu/Unofficial-DGNN-PyTorch 
"""


sets = {"val"}
datasets = {"kinetics"}  # if kinetics is used
# datasets = {'xsub', 'xview'}

for dataset, set in itertools.product(datasets, sets):
    print(dataset, set)
    data_jpt = np.load(f"{dataset}_data/{set}_data_joint.npy")
    print(len(data_jpt))
    data_bone = np.load(f"{dataset}_data/{set}_data_bone.npy")
    print(len(data_bone))
    N, C, T, V, M = data_jpt.shape
    data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)
    np.save(f"{dataset}_data/{set}_data_joint_bones.npy", data_jpt_bone)
