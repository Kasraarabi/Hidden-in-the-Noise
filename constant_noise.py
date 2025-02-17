import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *


def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['constant_noise'])
        wandb.config.update(args)
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)


    results = []


    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]
        
        # Generation for Image A
        set_random_seed(seed)
        initial_noise_A = pipe.get_random_latents()
        outputs_A = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=initial_noise_A,
        )
        orig_image_A = outputs_A.images[0]

        # Reverse img for Image A
        img_A = transform_img(orig_image_A).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_A = pipe.get_image_latents(img_A, sample=False)
        reversed_noise_A = pipe.forward_diffusion(
            latents=image_latents_A,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # Generation for Image B
        set_random_seed(seed + 1)  # Different seed for Image B
        current_prompt = dataset[i+1][prompt_key]
        initial_noise_B = pipe.get_random_latents()
        outputs_B = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=initial_noise_B,
        )
        orig_image_B = outputs_B.images[0]

        # Reverse img for Image B
        img_B = transform_img(orig_image_B).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_B = pipe.get_image_latents(img_B, sample=False)
        reversed_noise_B = pipe.forward_diffusion(
            latents=image_latents_B,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # Evaluation
        pixel_diff_A = evaluate_pixel_similarity(initial_noise_A, reversed_noise_A)
        pixel_diff_B = evaluate_pixel_similarity(initial_noise_B, reversed_noise_B)
        pixel_diff_inA_revB = evaluate_pixel_similarity(initial_noise_A, reversed_noise_B)
        pixel_diff_inB_revA = evaluate_pixel_similarity(initial_noise_B, reversed_noise_A)
        fourier_diff_A = evaluate_fourier_similarity(initial_noise_A, reversed_noise_A)
        fourier_diff_B = evaluate_fourier_similarity(initial_noise_B, reversed_noise_B)
        fourier_diff_inA_revB = evaluate_fourier_similarity(initial_noise_A, reversed_noise_B)
        fourier_diff_inB_revA = evaluate_fourier_similarity(initial_noise_B, reversed_noise_A)
        pixel_cosine_diff_A = evaluate_pixel_cosine_similarity(initial_noise_A, reversed_noise_A)
        pixel_cosine_diff_B = evaluate_pixel_cosine_similarity(initial_noise_B, reversed_noise_B)
        pixel_cosine_inA_revB = evaluate_pixel_cosine_similarity(initial_noise_A, reversed_noise_B)
        pixel_cosine_inB_revA = evaluate_pixel_cosine_similarity(initial_noise_B, reversed_noise_A)
        fourier_cosine_A = evaluate_fourier_cosine_similarity(initial_noise_A, reversed_noise_A)
        fourier_cosine_B = evaluate_fourier_cosine_similarity(initial_noise_B, reversed_noise_B)
        fourier_cosine_inA_revB = evaluate_fourier_cosine_similarity(initial_noise_A, reversed_noise_B)
        fourier_cosine_inB_revA = evaluate_fourier_cosine_similarity(initial_noise_B, reversed_noise_A)
        
        # Log results
        wandb.log({
            "pixel_diff_A": pixel_diff_A,
            "pixel_diff_B": pixel_diff_B,
            "pixel_diff_inA_revB": pixel_diff_inA_revB,
            "pixel_diff_inB_revA": pixel_diff_inB_revA,
        })
        wandb.log({
            "fourier_diff_A": fourier_diff_A,
            "fourier_diff_B": fourier_diff_B,
            "fourier_diff_inA_revB": fourier_diff_inA_revB,
            "fourier_diff_inB_revA": fourier_diff_inB_revA,
            "pixel_cosine_diff_A": pixel_cosine_diff_A,
            "pixel_cosine_diff_B": pixel_cosine_diff_B,
            "pixel_cosine_inA_revB": pixel_cosine_inA_revB,
            "pixel_cosine_inB_revA": pixel_cosine_inB_revA,
            "fourier_cosine_A": fourier_cosine_A,
            "fourier_cosine_B": fourier_cosine_B,
            "fourier_cosine_inA_revB": fourier_cosine_inA_revB,
            "fourier_cosine_inB_revA": fourier_cosine_inB_revA,
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)