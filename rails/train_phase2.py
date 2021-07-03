import os

import cv2
import tqdm
import numpy as np
import torch

from .rails import RAILS
from .datasets import data_loader
from .logger import Logger

import torchvision.transforms as transforms
from rails.network import Unet
from PIL import Image

class Resize_metrics(object):
    def __init__(self, w, h):
        self.size = (w, h)

    def __call__(self, sample):
        img = sample['image']
        img = img.resize((self.size[0], self.size[1]), resample=Image.BILINEAR)
        return {'image': img}

class ToTensor_metrics(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        img = sample['image']
        img = self.tensor(img).unsqueeze(0)
        return {'image': img}


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)) * 1.0)
    return 255.0 * norm_s_map


def blend_map(img, map, factor, colormap=cv2.COLORMAP_JET):
    assert 0 < factor < 1, 'factor must satisfy 0 < factor < 1'

    map = np.float32(map)
    map /= map.max()
    map *= 255
    map = map.astype(np.uint8)

    blend = cv2.addWeighted(src1=img, alpha=factor,
                            src2=cv2.applyColorMap(map, colormap), beta=(1 - factor),
                            gamma=0)

    return blend


#torch.set_printoptions(profile='full')

def main(args):
    
    rails = RAILS(args)
    data = data_loader('main', args)
    logger = Logger('carla_train_phase2', args)
    save_dir = logger.save_dir
    
    if args.resume:
        print ("Loading checkpoint from", args.resume)
        if rails.multi_gpu:
            rails.main_model.module.load_state_dict(torch.load(args.resume))
        else:
            rails.main_model.load_state_dict(torch.load(args.resume))
        start = int(args.resume.split('main_model_')[-1].split('.th')[0])
    else:
        start = 0

    global_it = 0

    model_dir = "/mnt/data/varun/sage_net/pretrained_models/picanet"
    models = sorted(os.listdir(model_dir), key=lambda x: int(x.split('epo_')[1].split('step')[0]))

    device = torch.device("cuda:2")

    # bdda_model = Unet().to(device)
    sage_model = Unet().to(device)

    print("Model loaded! Loading Checkpoint...")

    # bdda_model_name = models[0]
    sage_model_name = models[1]

    # bdda_state_dict = torch.load(os.path.join(model_dir, bdda_model_name))
    sage_state_dict = torch.load(os.path.join(model_dir, sage_model_name))

    # bdda_model.load_state_dict(bdda_state_dict)
    sage_model.load_state_dict(sage_state_dict)

    print("Checkpoint loaded! Now predicting...")

    # bdda_model.eval()
    sage_model.eval()

    print('==============================')

    for epoch in range(start,start+args.num_epoch):
        for wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds in tqdm.tqdm(data, desc='Epoch {}'.format(epoch)):
            annotated_rgb_imgs = []
            # print(f'b: {type(narr_rgbs), narr_rgbs.size()}')
            for img in narr_rgbs:
                img = img.numpy()
                img = Image.fromarray(img.astype('uint8'), 'RGB')
                # img.show()
                sample = {'image': img}
                w, h = img.size
                # print(f'w, h: {w, h}')
                transform = transforms.Compose([Resize_metrics(224, 224), ToTensor_metrics()])
                sample = transform(sample)

                img = sample['image'].to(device)

                with torch.no_grad():
                    sage_pred, _ = sage_model(img)

                sage_pred = sage_pred[5].data

                img, sage_pred = img[0].cpu().numpy(),  sage_pred[0, 0].cpu().numpy()

                im = np.transpose(img, (1, 2, 0))

                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                im = normalize_map(im)

                sage_pred = normalize_map(sage_pred)

                im = im.astype(np.uint8)

                sage_pred = sage_pred.astype(np.uint8)
                heatmap_sage = blend_map(im, sage_pred, factor=0.5)
                heatmap_sage = cv2.resize(heatmap_sage, (w, h))
                # print(heatmap_sage.shape)
                annotated_rgb_imgs.append(heatmap_sage)
                # convert to tensor
                # cv2.imshow('pred', sage_pred)
                # cv2.imshow('hm_sage_pred', heatmap_sage)
                # # cv2.imshow('ori', im)
                # cv2.waitKey(-1)

            annotated_rgb_imgs = np.array(annotated_rgb_imgs)
            annotated_rgb_imgs = torch.tensor(annotated_rgb_imgs)
            narr_rgbs = annotated_rgb_imgs
            # print(f'a: {type(narr_rgbs), narr_rgbs.size()}')
            #print(f'narr_rgbs : {narr_rgbs.shape}, ann_nar_rg :{type(annotated_rgb_imgs)}')
            return 0

            opt_info = rails.train_main(wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds)
            
            if global_it % args.num_per_log == 0:
                logger.log_main_info(global_it, opt_info)
        
            global_it += 1
    
        # Save model
        if (epoch+1) % args.num_per_save == 0:
            save_path = f'{save_dir}/main_model_{epoch+1}.th'
            torch.save(rails.main_model_state_dict(), save_path)
            print (f'saved to {save_path}')

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--resume', default=None)
    
    parser.add_argument('--data-dir', default='/mnt/data/varun/main_dataset')
    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    
    # Training data config
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--num-repeat', type=int, default=4)    # Should be consistent with autoagents/collector_agents/config.yaml

    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=3e-5)
    
    parser.add_argument('--num-per-log', type=int, default=100, help='per iter')
    parser.add_argument('--num-per-save', type=int, default=1, help='per epoch')
    
    parser.add_argument('--balanced-cmd', action='store_true')

    args = parser.parse_args()
    main(args)
