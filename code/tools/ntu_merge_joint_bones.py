import os
import numpy as np

'''
Function adapted from: https://github.com/kenziyuliu/Unofficial-DGNN-PyTorch 
'''


sets = {
    'val'
}

# 'ntu/xview', 'ntu/xsub', 'kinetics'
datasets = {
    'xsub'
}

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load('/multiverse/datasets/plizzari/Output_skeletons_without_missing_skeletons/{}/{}_data_joint_filtered.npy'.format(dataset, set))
        print(len(data_jpt))
        data_bone = np.load('/multiverse/datasets/plizzari/Output_skeletons_without_missing_skeletons/{}/{}_data_bones.npy'.format(dataset, set))
        print(len(data_bone))
        N, C, T, V, M = data_jpt.shape
        data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)
        np.save('/multiverse/datasets/plizzari/Output_skeletons_without_missing_skeletons/{}/{}_data_joint_bones.npy'.format(dataset, set), data_jpt_bone)