import os
import sys
import time
import math
import argparse
import numpy as np
from tqdm import tqdm
import torch
import json

from imageio import imwrite

sys.path.insert(0, '/home/ogallo/src-3rdparty/LiftedGAN/')

import utils
from models.lifted_gan import LiftedGAN


# __Angles Convention:__
# yaw, pitch
# ALL MOVEMENTS ARE IN PERSON-SPACE (ie left would be person moving head left, NOT observer)
# pitch:
# +-pi is camera-facing,
# slight up is -pi + eps, slight down is +pi - eps, straght up is -pi/2, straight down is +pi/2
# yaw:
# 0 is camera-facing, slight left is 0 -eps, slight right is 0 + eps, left is -pi/2, right is pi/2

class Poses():

    def __init__(self, file_name):
        with open(file_name, 'r') as f:
            self.pose_list = json.load(f)

    def get_pose(self, i):
        # Code from Eric:
        # pose = self.pose_list[i]
        # if pose[1] < 0: # face is looking up
        #     pose[1] += np.pi
        # if pose[1] > 0: # face is looking down
        #     pose[1] -= np.pi
        pose = self.pose_list[i]# - np.array([0, np.pi])
        return pose

    def get_angles(self, i):
        return np.degrees(self.get_pose(i))

    def get_average(self):
        avg = np.average(self.pose_list,0)
        return np.degrees(avg)

    def __len__(self):
        return len(self.pose_list)


def main(args):

    poses = Poses(args.pose_file)

    # for i in range(np.minimum(10, len(poses))):
    #     print(poses.get_angles(i))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = LiftedGAN()
    model.load_model(args.model)

    print('Forwarding the network...')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    b = 1 # Was batch size, but I decided to remove the batch.
    # num_batches = int(len(poses)/b);
    # for head in tqdm(range(0, args.n_samples, args.batch_size)):
    num_images = len(poses)
    if args.n_samples > 0:
        num_images = np.minimum(args.n_samples, num_images)

    for i in tqdm(range(num_images)):
        with torch.no_grad():
            # tail = min(args.n_samples, head+args.batch_size)
            # b = tail-head

            latent = torch.randn((b,512))
            styles = model.generator.style(latent)
            styles = args.truncation * styles + (1-args.truncation) * model.w_mu

            canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model.estimate(styles)

            view_rotate = view.clone()

            angles = poses.get_angles(i)
            
            yaw = angles[0] / model.xyz_rotation_range
            pitch = angles[1] / model.xyz_rotation_range

            view_rotate[:,0] = torch.ones(b) * pitch
            view_rotate[:,1] = torch.ones(b) * yaw
            view_rotate[:,2] = torch.ones(b) * 0
            view_rotate[:,3] = torch.sin(view_rotate[:,1]) * 0.1
            view_rotate[:,4] = - torch.sin(view_rotate[:,0]) * 0.2
            view_rotate[:,5] = torch.ones(b) * 0
            recon_rotate = model.render(canon_depth, canon_albedo, canon_light, view_rotate, trans_map=trans_map)[0]

            outputs = recon_rotate.permute(0,2,3,1).cpu().numpy() * 0.5 + 0.5
            outputs = np.minimum(1.0,np.maximum(0.0,outputs))
            outputs = (outputs*255).astype(np.uint8)

            for j in range(outputs.shape[0]):
                imwrite(f'{args.output_dir}/{i*b+(j+1):05d}_yaw_{int(angles[0]):02d}_pitch_{int(angles[1]):02d}.png', outputs[j])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="The path to the pre-trained model",
                        type=str)
    parser.add_argument("--output_dir", help="The output path",
                        type=str)
    parser.add_argument("--truncation", help="Truncation of latent styles",
                        type=int, default=0.7)
    parser.add_argument("--n_samples", help="Number of images to generate (0 for no limit)",
                        type=int, default=0)
    parser.add_argument("--pose_file", help="The path to the JSON file with the poses to generate.",
                        type=str, default='angles.json')
    args = parser.parse_args()
    
    main(args)
