from tqdm import tqdm
import torch
import itertools
import argparse
import os
from datetime import datetime
import pandas as pd
from collections import OrderedDict
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
import wandb
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
import hashlib  
import itertools
from copy import deepcopy
from scipy.ndimage import rotate
from PIL import Image
import numpy as np
from utils import *
from io_utils import *
import torch.nn.functional as F
from collections import Counter
import random
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='multiple-key identification with noise matching')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--reference_model', default='ViT-g-14')
    parser.add_argument('--reference_model_pretrain', default='laion2b_s12b_b42k')
    parser.add_argument('--online', action='store_true', default=False, help='True to check cache and download models if necessary. False to use cached models.')

    group = parser.add_argument_group('hyperparameters')
    parser.add_argument('--general_seed', type=int, default=42)
    parser.add_argument('--watermark_seed', type=int, default=5)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--ring_width', default=1, type=int)
    parser.add_argument('--num_inmost_keys', default=2, type=int)
    parser.add_argument('--ring_value_range', default=64, type=int)
    
    parser.add_argument('--save_generated_imgs', type=bool, default=False)
    parser.add_argument('--save_root_dir', type=str, default='./runs')

    group = parser.add_argument_group('trials parameters')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--trials', type=int, default=100, help='total number of trials to run')
    parser.add_argument('--fix_gt', type=int, default=1, help='use watermark after discarding the imag part on space domain as gt.')
    parser.add_argument('--time_shift', type=int, default=1, help='use time-shift')
    parser.add_argument('--time_shift_factor', type=float, default=1.0, help='factor to scale the value after time-shift')
    parser.add_argument('--assigned_keys', type=int, default=-1, help='number of assigned keys, -1 for all possible keys')
    parser.add_argument('--channel_min', type=int, default=1, help='only for heterogeneous watermark, when match gt, take min among channels as the result')

    # New arguments for noise matching
    parser.add_argument('--M', type=int, default=10000, help='number of background noises')
    parser.add_argument('--wandb_project', type=str, default='watermark_identification', help='Weights & Biases project name')

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps

    return args

def main(args):

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'


    timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(args.save_root_dir, timestr + '_' + args.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    if args.save_generated_imgs:
        save_img_dir = os.path.join(save_dir, 'images', 'watermarked')
        os.makedirs(save_img_dir, exist_ok=False)
        save_nowatermark_img_dir = os.path.join(save_dir, 'images', 'no_watermark')
        os.makedirs(save_nowatermark_img_dir, exist_ok=False)
    
    set_random_seed(args.general_seed)

    # Load model
    username = os.getlogin()
    if args.online:
        pipeline_pretrain = args.model_id
        reference_model_pretrain = args.reference_model_pretrain
        dataset_id = 'Gustavosta/Stable-Diffusion-Prompts'
    else:
        pipeline_pretrain = f'{os.path.expanduser("~")}/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741/'
        reference_model_pretrain = f'{os.path.expanduser("~")}/.cache/huggingface/hub/models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K/snapshots/4b0305adc6802b2632e11cbe6606a9bdd43d35c9/open_clip_pytorch_model.bin'
        dataset_id = 'Gustavosta/stable-diffusion-prompts'
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    model_dtype = torch.float16
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler', local_files_only=(not args.online))
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        pipeline_pretrain,
        scheduler=scheduler,
        torch_dtype=model_dtype,
        revision='fp16',
        )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model, 
            pretrained=reference_model_pretrain, 
            device=device
            )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    dataset, prompt_key = get_dataset(dataset_id)

    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    base_latents = pipe.get_random_latents()
    base_latents = base_latents.to(torch.float64)
    original_latents_shape = base_latents.shape
    sing_channel_ring_watermark_mask = torch.tensor(
            ring_mask(
                size = original_latents_shape[-1], 
                r_out = RADIUS, 
                r_in = RADIUS_CUTOFF)
            )
    
    if len(HETER_WATERMARK_CHANNEL) > 0:
        single_channel_heter_watermark_mask = torch.tensor(
                ring_mask(
                    size = original_latents_shape[-1], 
                    r_out = RADIUS, 
                    r_in = RADIUS_CUTOFF)
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
    key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-args.ring_value_range, args.ring_value_range, args.num_inmost_keys).tolist(), repeat = len(RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
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
            Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)))
    

    watermarker = WatermarkClass(args, device, args.M, ring_capacity, Fourier_watermark_pattern_list, watermark_region_mask, pipe)

    results_list = []
    score_heads = ['CLIP No Watermark', 'CLIP Fourier Watermark']
    quality_metrics = QualityResultsCollector(score_heads)

    true_scores = []
    random_scores = []

    wandb.init(project="Paper_Results", config=args.__dict__, tags=["fast"])
    wandb.run.name = f"{args.M}_{ring_capacity}_{args.model_id}_new"
    wandb.run.save()
    
    wandb.run.log_code('identify_new_rotation_resistant.py')

    wandb.config.update({
        "model_id": args.model_id,
        "reference_model": args.reference_model,
        "ring_capacity": ring_capacity,
        "RADIUS": RADIUS,
        "RADIUS_CUTOFF": RADIUS_CUTOFF,
        "WATERMARK_CHANNEL": WATERMARK_CHANNEL,
        "RING_WATERMARK_CHANNEL": RING_WATERMARK_CHANNEL,
        "HETER_WATERMARK_CHANNEL": HETER_WATERMARK_CHANNEL,
    })

    for prompt_index in tqdm(range(args.trials)):
        this_seed = args.general_seed + prompt_index
        this_prompt = dataset[prompt_index][prompt_key]

        set_random_seed(this_seed)
 
        Fourier_watermark_image, true_n, true_m, Fourier_watermark_latents = watermarker.encode_and_generate(pipe, this_prompt, prompt_index)

        no_watermark_latents = pipe.get_random_latents()
        no_watermark_image = pipe(
            this_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=no_watermark_latents
        ).images[0]

        no_watermark_clip, Fourier_watermark_clip = measure_similarity([no_watermark_image, Fourier_watermark_image], this_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
        quality_metrics.collect('CLIP No Watermark', no_watermark_clip.item())
        quality_metrics.collect('CLIP Fourier Watermark', Fourier_watermark_clip.item())

        if args.save_generated_imgs:
            Fourier_watermark_image.save(os.path.join(save_img_dir, f'Key_{key_index}.Prompt_{prompt_index}.Fourier_watermark.ClipSim_{Fourier_watermark_clip.item():.4f}.jpg'))
            no_watermark_image.save(os.path.join(save_nowatermark_img_dir, f'Key_{key_index}.Prompt_{prompt_index}.No_watermark.ClipSim_{no_watermark_clip.item():.4f}.jpg'))
        

        distorted_image_list = [
            [None, Fourier_watermark_image],
            image_distortion(None, Fourier_watermark_image, seed = this_seed, r_degree = 75),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, jpeg_ratio = 25),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, crop_scale = 0.75, crop_ratio = 0.75),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, gaussian_blur_r = 8),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, gaussian_std = 0.1),
            image_distortion(None, Fourier_watermark_image, seed = this_seed, brightness_factor = 6),
        ]
        head = ['Clean', 'Rot 75', 'JPEG 25', 'C&S 75', 'Blur 8', 'Noise 0.1', 'Brightness [0, 6]', 'Avg']

        this_it_results = []

        for distorted_image in distorted_image_list:
            detected_n, detected_m, true_distance, random_distances = watermarker.detect_and_evaluate(pipe, distorted_image[1])
            acc_key = detected_n == true_n
            acc_noise = detected_m == true_m
            this_it_results.append(acc_noise)
            true_scores.append(true_distance)
            random_scores.extend(random_distances)
            
            print(f"Distortion: {head[distorted_image_list.index(distorted_image)]}, Detected n: {detected_n}, True n: {true_n}, Detected m: {detected_m}, True m: {true_m}")
            print(f"Accuracy - Key: {acc_key}, Noise: {acc_noise}")

        results_list.append(this_it_results)

        wandb.log({
            "iteration": prompt_index,
            "no_watermark_clip": no_watermark_clip.item(),
            "Fourier_watermark_clip": Fourier_watermark_clip.item(),
        })

        for i, (distortion, acc_noise) in enumerate(zip(head[:-1], this_it_results)):
            wandb.log({
                f"accuracy_noise_{distortion}": acc_noise,
                "iteration": prompt_index,
            })

        wandb.log({
            "true_distance": true_distance,
            "random_distance_mean": np.mean(random_distances),
            "random_distance_std": np.std(random_distances),
            "iteration": prompt_index,
        })

    print('-' * 40)

    print(f'Ring capacity = {ring_capacity}')

    result_array = np.mean(results_list, axis=0)
    result_array_avg = np.mean(result_array)

    table = PrettyTable()
    table.field_names = ["Accuracy"] + head

    row = ["Noise Detection"]
    for it in range(len(result_array)):
        row.append(f'{result_array[it]:.3f}')
    row.append(f'{result_array_avg:.3f}')
    table.add_row(row)
    print(table)

    print()
    quality_metrics.print_average()



    # Ensure both true_scores and random_scores are tensors or lists of numbers
    true_scores = [score.cpu().item() if isinstance(score, torch.Tensor) else score for score in true_scores]
    random_scores = [score.cpu().item() if isinstance(score, torch.Tensor) else score for score in random_scores]


    for distortion, acc_noise in zip(head, row[1:]):
        wandb.log({f"final_accuracy_noise_{distortion}": acc_noise})

    for metric, value in quality_metrics.return_average().items():
        wandb.log({f"final_{metric}": value})

    table = wandb.Table(columns=["Metric"] + head)
    table.add_data("Noise Detection", *row[1:])
    wandb.log({"results_table": table})

    if args.save_generated_imgs:
        wandb.log({"watermarked_image": wandb.Image(Fourier_watermark_image, caption=f"Key_{key_index}.Prompt_{prompt_index}.Fourier_watermark")})
        wandb.log({"no_watermark_image": wandb.Image(no_watermark_image, caption=f"Key_{key_index}.Prompt_{prompt_index}.No_watermark")})


    df_exp = pd.DataFrame([row[1:]], columns=head)

    quality_scores = quality_metrics.return_average()
    df_qual = pd.DataFrame({k: [v] for k, v in quality_scores.items()})

    df_hyper = pd.DataFrame(
        OrderedDict(
            [
                ('User', [username]),
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
                ('M', [args.M])            
                ]
        )
    )
    df = pd.concat([df_hyper, df_exp, df_qual], axis=1)
    df.to_csv(os.path.join(save_dir, 'log.csv'), index=False, float_format="%.3f")



class WatermarkClass:
    def __init__(self, args, device, M, N, Fourier_watermark_pattern_list, watermark_region_mask, pipe):
        self.args = args
        self.device = device
        self.M = M
        self.N = N
        self.Fourier_watermark_pattern_list = Fourier_watermark_pattern_list
        self.watermark_region_mask = watermark_region_mask
        self.watermarked_latents = {}
        self.salt = "ekofijorfgjirejoiconime"
        self.text_embeddings = pipe.get_text_embedding('')


    def _generate_hash(self, m):
        n = m % self.N
        hash_input = f"{m}{self.salt}{n}{(m % n + 1) if n != 0 else random.randint(1, 100)}".encode('utf-8')
        hash_object = hashlib.sha256(hash_input)
        hash_output = int.from_bytes(hash_object.digest(), byteorder='big') % (2**32 - 1)
        return hash_output, n


    def encode_and_generate(self, pipe, prompt, iteration):
        m = random.randint(1, self.M)
        hash_value, n = self._generate_hash(m)
        
        # Use hash as seed to generate noise
        torch.manual_seed(hash_value)
        torch.cuda.manual_seed_all(hash_value)
        noise = torch.randn(4, 64, 64, device=self.device, dtype=torch.float16)
        
        # Add the n-th key/ring pattern to the noise
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
        
        with torch.no_grad():
            image = pipe(prompt, latents=encoded_noise, num_inference_steps=self.args.num_inference_steps).images[0]
        return image, n, m, encoded_noise


        
    def detect_watermark_stage1(self, pipe, image):
        # Extract latent representation
        image_tensor = transform_img(image).unsqueeze(0).to(device=self.device, dtype=pipe.vae.dtype)
        image_latents = pipe.get_image_latents(image_tensor, sample=False)

        # Reverse diffusion process
        reversed_latent = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=self.text_embeddings,
            guidance_scale=1,
            num_inference_steps=self.args.test_num_inference_steps,
        )

        # Identify the correct ring (key)
        min_key_distance = float('inf')
        detected_n = -1
        reversed_latent_fft = fft(reversed_latent)

        for n in range(self.N):
            key_pattern = self.Fourier_watermark_pattern_list[n]
            
            distance = get_distance(
                key_pattern, 
                reversed_latent_fft, 
                self.watermark_region_mask,
                p=2,  # Using L2 norm
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
        rotation_angles = [0,74,76] # The method is robust to any degree of rotation by checking rotation patterns like 2n or 4n+1.

        # Search among the group with the same ring (key)
        for (n, m), watermarked_latent in self.watermarked_latents.items():
            if n == detected_n:
                watermarked_latent = watermarked_latent.to(self.device)
                for angle in rotation_angles:
                    watermarked_latent = transforms.RandomRotation((angle, angle))(watermarked_latent)
                    distance = torch.norm(reversed_latent - watermarked_latent)
                    if distance < min_distance:
                        min_distance = distance
                        detected_m = m


        true_distance = min_distance

        # Random distances based on L2 norms
        random_distances = [torch.norm(reversed_latent - torch.randn_like(reversed_latent)).item() for _ in range(100)]


        return detected_n, detected_m, true_distance, random_distances




if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        wandb.finish(exit_code=1)
        raise e
