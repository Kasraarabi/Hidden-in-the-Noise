import argparse
import os
import random
import json
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm
import itertools
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from prettytable import PrettyTable
import wandb
from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from utils import *
from io_utils import *

'''
Before using this code, please set RING_WATERMARK_CHANNEL to 0 in utils.py: RING_WATERMARK_CHANNEL = [0]
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Inpainting-based watermarking for existing images')
    parser.add_argument('--run_name', default='inpaint_watermark')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-inpainting')
    parser.add_argument('--online', action='store_true', default=False)
    
    parser.add_argument('--meta_data_path', type=str, default='fid_outputs/coco/meta_data.json', help='Path to meta_data.json file')
    parser.add_argument('--ground_truth_folder', type=str, default='fid_outputs/coco/ground_truth', help='Path to folder containing ground truth images')
    parser.add_argument('--mask_size', type=float, default=0.5, help='Size of the inpainting mask as a fraction of image size')
    
    parser.add_argument('--general_seed', type=int, default=42)
    parser.add_argument('--watermark_seed', type=int, default=5)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--ring_width', default=1, type=int)
    parser.add_argument('--num_inmost_keys', default=2, type=int)
    parser.add_argument('--ring_value_range', default=64, type=int)
    
    parser.add_argument('--save_watermarked_imgs', type=bool, default=False)
    parser.add_argument('--save_root_dir', type=str, default='./inpaint_runs')
    
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--trials', type=int, default=1000, help='total number of trials to run')
    parser.add_argument('--fix_gt', type=int, default=1)
    parser.add_argument('--time_shift', type=int, default=1)
    parser.add_argument('--time_shift_factor', type=float, default=1.0)
    parser.add_argument('--assigned_keys', type=int, default=-1)
    parser.add_argument('--channel_min', type=int, default=1)
    
    parser.add_argument('--M', type=int, default=10000, help='number of background noises')
    parser.add_argument('--wandb_project', type=str, default='inpaint_watermark_identification')
    parser.add_argument('--fid_threshold', type=float, default=20.0, help='FID threshold for logging images')
    parser.add_argument('--mask_size_search', nargs='+', type=float, default=[0.1], help='Mask sizes to search over')
    
    args = parser.parse_args()
    return args

class ImageDataset(Dataset):
    def __init__(self, meta_data_path, ground_truth_folder, transform=None):
        with open(meta_data_path, 'r') as f:
            self.meta_data = json.load(f)['images']
        self.ground_truth_folder = ground_truth_folder
        self.transform = transform

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_info = self.meta_data[idx]
        img_path = os.path.join(self.ground_truth_folder, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_info['file_name'], img_info.get('caption', '')

class InpaintWatermarker:
    def __init__(self, args, device, M, N, Fourier_watermark_pattern_list, watermark_region_mask, pipe):
        self.args = args
        self.device = device
        self.M = M
        self.N = N
        self.Fourier_watermark_pattern_list = Fourier_watermark_pattern_list
        self.watermark_region_mask = watermark_region_mask
        self.watermarked_latents = {}
        self.salt = "ekofijorfgjirejoiconime"
        # self.text_embeddings = pipe.get_text_embedding('')
        
        self.mask = self._create_inpaint_mask()
        self.latent_mask = self._create_latent_mask(pipe)

    def _create_inpaint_mask(self):
        mask_size = int(self.args.image_size * self.args.mask_size)
        start = (self.args.image_size - mask_size) // 2
        mask = np.zeros((self.args.image_size, self.args.image_size), dtype=np.float32)
        mask[start:start+mask_size, start:start+mask_size] = 1
        return mask

    def _create_latent_mask(self, pipe):
        latent_size = self.args.image_size // pipe.vae_scale_factor
        mask_size = int(latent_size * self.args.mask_size)
        start = (latent_size - mask_size) // 2
        latent_mask = torch.zeros((1, 1, latent_size, latent_size), device=self.device)
        latent_mask[:, :, start:start+mask_size, start:start+mask_size] = 1
        return latent_mask

    def _generate_hash(self, m):
        n = m % self.N
        hash_input = f"{m}{self.salt}{n}{m%n}".encode('utf-8')
        hash_object = hashlib.sha256(hash_input)
        hash_output = int.from_bytes(hash_object.digest(), byteorder='big') % (2**32 - 1)
        return hash_output, n

    def encode_and_generate(self, pipe, image, iteration):
        m = random.randint(1, self.M)
        hash_value, n = self._generate_hash(m)
        
        torch.manual_seed(hash_value)
        torch.cuda.manual_seed_all(hash_value)
        noise = torch.randn(4, 64, 64, device=self.device, dtype=torch.float16)
        
        Fourier_watermark_latents = generate_Fourier_watermark_latents(
            device=self.device,
            radius=RADIUS,
            radius_cutoff=RADIUS_CUTOFF,
            original_latents=noise.unsqueeze(0),
            watermark_pattern=self.Fourier_watermark_pattern_list[n],
            watermark_channel=WATERMARK_CHANNEL,
            watermark_region_mask=self.watermark_region_mask,
        )
        
        encoded_noise = Fourier_watermark_latents.to(dtype=pipe.unet.dtype, device=pipe.device)
        
        self.watermarked_latents[(n, m)] = encoded_noise.cpu()
        
        image = image.to(device=pipe.device)
        mask = torch.from_numpy(self.mask).to(device=pipe.device)
        
        with torch.no_grad():
            watermarked_image = pipe(
                prompt="",
                image=image,
                mask_image=mask,
                latents=encoded_noise,
                num_inference_steps=self.args.num_inference_steps,
                guidance_scale=self.args.guidance_scale
            ).images[0]
        
        if not isinstance(watermarked_image, Image.Image):
            watermarked_image = Image.fromarray((watermarked_image * 255).astype(np.uint8))
        
        return watermarked_image, n, m, encoded_noise

    def detect_watermark_stage1(self, pipe, image):
        image_tensor = transform_img(image).unsqueeze(0).to(device=self.device, dtype=pipe.vae.dtype)
        image_latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        image_latents = image_latents * pipe.vae.config.scaling_factor
        
        masked_image_latents = image_latents * (1 - self.latent_mask) + self.latent_mask * torch.randn_like(image_latents)
        
        reversed_latent = pipe.scheduler.add_noise(
            masked_image_latents,
            torch.randn_like(masked_image_latents),
            torch.tensor([self.args.num_inference_steps]).to(self.device)
        )

        min_key_distance = float('inf')
        detected_n = -1
        reversed_latent_fft = fft(reversed_latent)

        for n in range(self.N):
            key_pattern = self.Fourier_watermark_pattern_list[n]
            
            distance = get_distance(
                key_pattern, 
                reversed_latent_fft, 
                self.watermark_region_mask,
                p=2,
                mode='complex',
                channel_min=self.args.channel_min,
                channel=WATERMARK_CHANNEL
            )
            
            if distance < min_key_distance:
                min_key_distance = distance
                detected_n = n

        return detected_n, reversed_latent

    def detect_and_evaluate(self, pipe, image):
        detected_n, reversed_latent = self.detect_watermark_stage1(pipe, image)

        min_distance = float('inf')
        detected_m = -1

        for (n, m), watermarked_latent in self.watermarked_latents.items():
            if n == detected_n:
                watermarked_latent = watermarked_latent.to(self.device)
                distance = torch.norm(reversed_latent - watermarked_latent)
                if distance < min_distance:
                    min_distance = distance
                    detected_m = m

        if detected_m == -1:
            for (n, m), watermarked_latent in self.watermarked_latents.items():
                watermarked_latent = watermarked_latent.to(self.device)
                distance = torch.norm(reversed_latent - watermarked_latent)
                if distance < min_distance:
                    min_distance = distance
                    detected_n = n
                    detected_m = m

        true_distance = min_distance

        random_distances = [torch.norm(reversed_latent - torch.randn_like(reversed_latent)).item() for _ in range(100)]

        return detected_n, detected_m, true_distance, random_distances


def calculate_fid(original_image, watermarked_image, device):
    fid = FrechetInceptionDistance().to(device)
    
    def preprocess_image(image):
        # Ensure image is in the range [0, 255] and convert to uint8
        image = (image * 255).clamp(0, 255).to(torch.uint8)
        
        # If image is not in the format [B, C, H, W], rearrange it
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension if missing
        if image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        
        return image
    
    original_image = preprocess_image(original_image)
    watermarked_image = preprocess_image(watermarked_image)
    
    fid.update(original_image, real=True)
    fid.update(watermarked_image, real=False)
    return fid.compute().item()

def calculate_inception_score(images, device):
    device = torch.device('cuda')
    inception = InceptionScore().to(device)
    images = images.to(device).to(torch.uint8)
    inception.update(images)
    return inception.compute()[0].item()

def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(args.save_root_dir, timestr + '_' + args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    if args.save_watermarked_imgs:
        save_img_dir = os.path.join(save_dir, 'watermarked_images')
        os.makedirs(save_img_dir, exist_ok=True)
    
    set_random_seed(args.general_seed)

    model_dtype = torch.float16
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler', local_files_only=(not args.online))
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=model_dtype,
        revision='fp16',
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = ImageDataset(args.meta_data_path, args.ground_truth_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    base_latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float64)
    original_latents_shape = base_latents.shape
    sing_channel_ring_watermark_mask = torch.tensor(
        ring_mask(
            size=original_latents_shape[-1], 
            r_out=RADIUS, 
            r_in=RADIUS_CUTOFF)
    )
    
    if len(HETER_WATERMARK_CHANNEL) > 0:
        single_channel_heter_watermark_mask = torch.tensor(
            ring_mask(
                size=original_latents_shape[-1], 
                r_out=RADIUS, 
                r_in=RADIUS_CUTOFF)
        )
        heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1).to(device)

    watermark_region_mask = []
    for channel_idx in WATERMARK_CHANNEL:
        if channel_idx in RING_WATERMARK_CHANNEL:
            watermark_region_mask.append(sing_channel_ring_watermark_mask)
        else:
            watermark_region_mask.append(single_channel_heter_watermark_mask)
    watermark_region_mask = torch.stack(watermark_region_mask).to(device)

    single_channel_num_slots = RADIUS - RADIUS_CUTOFF
    key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-args.ring_value_range, args.ring_value_range, args.num_inmost_keys).tolist(), repeat=len(RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
    key_value_combinations = list(itertools.product(*key_value_list))

    if args.assigned_keys > 0:
        assert args.assigned_keys <= len(key_value_combinations)
        key_value_combinations = random.sample(key_value_combinations, k=args.assigned_keys)


    Fourier_watermark_pattern_list = [make_Fourier_ringid_pattern(device, list(combo), base_latents, 
                                                                  radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
                                                                  ring_watermark_channel=RING_WATERMARK_CHANNEL, 
                                                                  heter_watermark_channel=HETER_WATERMARK_CHANNEL,
                                                                  heter_watermark_region_mask=heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL)>0 else None)
                                      for _, combo in enumerate(key_value_combinations)]            

    ring_capacity = len(Fourier_watermark_pattern_list)

    print(f'[Info] Ring capacity = {ring_capacity}')

    if args.fix_gt:
        Fourier_watermark_pattern_list = [fft(ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in Fourier_watermark_pattern_list]
    
    if args.time_shift:
        for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
            Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim=(-1, -2)))

    results_list = []
        
    wandb.init(project=args.wandb_project, config=args.__dict__, tags=["inpaint_watermark"])
    wandb.run.name = f"{timestr}_{args.run_name}"
    wandb.run.save()

    wandb.run.log_code('inpaint_watermark.py')

    wandb.config.update({
        "model_id": args.model_id,
        "ring_capacity": ring_capacity,
        "RADIUS": RADIUS,
        "RADIUS_CUTOFF": RADIUS_CUTOFF,
        "WATERMARK_CHANNEL": WATERMARK_CHANNEL,
        "RING_WATERMARK_CHANNEL": RING_WATERMARK_CHANNEL,
        "HETER_WATERMARK_CHANNEL": HETER_WATERMARK_CHANNEL,
    })

    best_fid = float('inf')
    best_mask_size = None
    best_images = []

    fid = FrechetInceptionDistance(feature=64).to(device)
    highest_fid_images = []

    for mask_size in args.mask_size_search:
        args.mask_size = mask_size
        watermarker = InpaintWatermarker(args, device, args.M, len(Fourier_watermark_pattern_list), Fourier_watermark_pattern_list, watermark_region_mask, pipe)        

        batch_original = []
        batch_watermarked = []
        batch_names = []
        batch_captions = []

        for prompt_index, (image, image_name, caption) in tqdm(enumerate(dataloader)):
            if prompt_index >= args.trials:
                break

            this_seed = args.general_seed + prompt_index
            set_random_seed(this_seed)
            m_index = random.randint(0, args.M - 1)
            key_index = random.randint(0, ring_capacity - 1)

            image = image.squeeze(0).to(device)
            watermarked_image, encoded_n, m, watermarked_latents = watermarker.encode_and_generate(pipe, image, prompt_index)

            if args.save_watermarked_imgs:
                watermarked_image.save(os.path.join(save_img_dir, f'Key_{key_index}.Image_{image_name[0]}.Watermarked.jpg'))

            # Convert image tensors to uint8 format
            original_uint8 = (image.squeeze(0) * 255).byte()
            watermarked_uint8 = (transforms.ToTensor()(watermarked_image) * 255).byte()
            
            batch_original.append(original_uint8)
            batch_watermarked.append(watermarked_uint8)
            batch_names.append(image_name[0] if isinstance(image_name, (list, tuple)) else image_name)
            batch_captions.append(caption[0] if isinstance(caption, (list, tuple)) else caption)

            if len(batch_original) == 32 or prompt_index == args.trials - 1:
                batch_original_tensor = torch.stack(batch_original).to(device)
                batch_watermarked_tensor = torch.stack(batch_watermarked).to(device)

                fid.update(batch_original_tensor, real=True)
                fid.update(batch_watermarked_tensor, real=False)

                # Convert back to float for inception score calculation
                batch_watermarked_float = batch_watermarked_tensor.float() / 255.0
                inception_score = calculate_inception_score(batch_watermarked_float, device)

                for i in range(len(batch_original)):
                    highest_fid_images.append({
                        'original': batch_original[i].float() / 255.0,
                        'watermarked': batch_watermarked[i].float() / 255.0,
                        'name': batch_names[i],
                        'caption': batch_captions[i],
                        'inception_score': inception_score,
                        'mask_size': mask_size
                    })

                batch_original = []
                batch_watermarked = []
                batch_names = []
                batch_captions = []

        fid_score = fid.compute().item()
        print(f"FID score for mask size {mask_size}: {fid_score}")
        wandb.log({f"fid_score_{mask_size}": fid_score})

        fid.reset()

    # Sort highest_fid_images by inception score (higher is worse) and keep top 50
    highest_fid_images = sorted(highest_fid_images, key=lambda x: x['inception_score'], reverse=True)[:50]

    # Log the top 50 highest FID images
    for i, img_data in enumerate(highest_fid_images):
        wandb.log({
            f"top_{i+1}_fid_pair": [
                wandb.Image(img_data['original'].squeeze().permute(1, 2, 0).cpu().numpy(), caption=f"Original: {img_data['name']}"),
                wandb.Image(img_data['watermarked'].squeeze().permute(1, 2, 0).cpu().numpy(), caption=f"Watermarked: {img_data['name']}"),
            ],
            f"top_{i+1}_fid_caption": img_data['caption'],
            f"top_{i+1}_inception_score": img_data['inception_score'],
            f"top_{i+1}_fid_mask_size": img_data['mask_size']
        })

    best_fid = best_images[0]['fid_score'] if best_images else float('inf')
    best_mask_size = best_images[0]['mask_size'] if best_images else None

    print(f'Best mask size: {best_mask_size}, Best FID: {best_fid}')
    wandb.log({
        "best_mask_size": best_mask_size,
        "best_fid": best_fid
    })

    wandb.finish()

    df_exp = pd.DataFrame([{
        'Best Mask Size': best_mask_size,
        'Best FID': best_fid
    }])

    df_hyper = pd.DataFrame(
        OrderedDict(
            [
                ('User', [os.getlogin()]),
                ('Date', [datetime.now().strftime('%Y.%m.%d')]),
                ('R_out', [RADIUS]),
                ('R_in', [RADIUS_CUTOFF]),
                ('Assigned keys', [ring_capacity]),
                ('Heter channels', [HETER_WATERMARK_CHANNEL]),
                ('Ring channels', [RING_WATERMARK_CHANNEL]),
                ('Shift factor', [args.time_shift_factor]),
                ('FixGT', ['TRUE' if args.fix_gt else 'FALSE']),
                ('Centroid', ['TRUE' if ANCHOR_Y_OFFSET == 0 else 'FALSE']),
                ('Time shift', ['TRUE' if args.time_shift else 'FALSE']),
                ('Trials', [args.trials]),
                ('M', [args.M]),
            ]
        )
    )
    df = pd.concat([df_hyper, df_exp], axis=1)
    df.to_csv(os.path.join(save_dir, 'log.csv'), index=False, float_format="%.3f")



if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        wandb.finish(exit_code=1)
        raise e
