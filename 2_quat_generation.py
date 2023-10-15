import os
import numpy as np
import torch
from tqdm import tqdm
from os.path import join as pjoin

from utils.skeleton import Skeleton
from utils.quaternion import *
from utils.paramUtil import *


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    return quat_params

'''
MAIN
'''

if __name__ == "__main__":

    # Insert the input, output path and file name
      
    data_dir = './output/positions_data/'
    save_dir = './output/quat_params_data/'

    file_name = 'example'
     
    l_idx1, l_idx2 = 5, 8               # Lower legs
    fid_r, fid_l = [8, 11], [7, 10]     # Right/Left foot
    face_joint_indx = [2, 1, 17, 16]    # Face direction, RHip, LHip, sdr_r, sdr_l    
    r_hip, l_hip = 2, 1                 # LHip, RHip
    
    joints_num = 22
     
    os.makedirs(save_dir, exist_ok=True)

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Get offsets of target skeleton
    input = np.load(os.path.join(data_dir, file_name + '.npy'))
    input = input.squeeze(0)
    input = input.reshape(len(input), -1, 3)
    input = torch.from_numpy(input)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(input[0])

    input = input.numpy()    
    
    # Get the quat_params
    quat_params = uniform_skeleton(input, tgt_offsets)

    np.save(save_dir+file_name+"_quat_params.npy",quat_params)

    print("Successfully Generated Quaternions!!!")