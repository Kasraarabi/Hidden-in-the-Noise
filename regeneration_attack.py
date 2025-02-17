import argparse
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from io_utils import *
from optim_utils import *
from PIL import Image
import os
from wmattacker import DiffWMAttacker
from att_src.diffusers.pipelines.stable_diffusion.pipeline_re_sd import ReSDPipeline

def cosine_similarity(a, b):
    return nn.functional.cosine_similarity(a.view(-1), b.view(-1), dim=0)

def l1_similarity(a, b):
    return torch.abs(a - b).sum()

def l2_similarity(a, b):
    return torch.norm(a - b, p=2)

def calculate_noise_similarities(target_noise, n_samples, pipe):
    cosine_similarities = []
    l1_similarities = []
    l2_similarities = []
    for _ in range(n_samples):
        random_noise = pipe.get_random_latents()
        cosine_similarities.append(cosine_similarity(target_noise, random_noise).item())
        l1_similarities.append(l1_similarity(target_noise, random_noise).item())
        l2_similarities.append(l2_similarity(target_noise, random_noise).item())
    return cosine_similarities, l1_similarities, l2_similarities

def plot_histograms(random_data, true_data, attacked_data, title, filename):
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", 3)
    
    for i, (data, label) in enumerate(zip([random_data, true_data, attacked_data], ['Random Noise', 'True Noise', 'Attacked Noise'])):
        plt.figure(figsize=(12, 8))
        sns.histplot(data, kde=True, stat="density", color=colors[i])
        
        mean_value = np.mean(data)
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.4f}')

        plt.xlabel('Similarity', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=14)
        
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.close()

    # Combine all plots into a single figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 24))
    
    for i, (data, label) in enumerate(zip([random_data, true_data, attacked_data], ['Random Noise', 'True Noise', 'Attacked Noise'])):
        sns.histplot(data, kde=True, stat="density", color=colors[i], ax=axes[i])
        
        mean_value = np.mean(data)
        axes[i].axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label='Mean')
        
        axes[i].set_title(f"{label}: {title}\nMean: {mean_value:.4f}", fontsize=20, fontweight='bold')
        axes[i].set_xlabel('Similarity', fontsize=16)
        axes[i].set_ylabel('Density', fontsize=16)
        axes[i].legend(fontsize=14)
        
        axes[i].tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f"combined_{filename}", dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    wandb.init(project='Paper_Results', name='Noise_Match_10000_with_Attack', tags=['noise_match', 'attack'])
    wandb.config.update(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
    )
    pipe = pipe.to(device)

    dataset, prompt_key = get_dataset(args)

    all_cosine_similarities = []
    all_l1_similarities = []
    all_l2_similarities = []
    all_initial_cosine = []
    all_initial_l1 = []
    all_initial_l2 = []
    all_attacked_cosine = []
    all_attacked_l1 = []
    all_attacked_l2 = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        original_prompt = dataset[i][prompt_key]
                    
        set_random_seed(seed)
        
        initial_noise = pipe.get_random_latents()
        
        outputs = pipe(
            original_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=initial_noise,
        )
        generated_image = outputs.images[0]

        # Process original image
        img = transform_img(generated_image).unsqueeze(0).to(initial_noise.dtype).to(device)
        image_latents = pipe.get_image_latents(img, sample=False)
        reversed_noise = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=pipe.get_text_embedding(""),
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # Apply attack
        attacked_image, attacked_latents = apply_attack(generated_image, 'diff_attacker_60', args.attack_params)
        
        # Process attacked image
        attacked_img = transform_img(attacked_image).unsqueeze(0).to(initial_noise.dtype).to(device)
        attacked_image_latents = pipe.get_image_latents(attacked_img, sample=False)
        attacked_reversed_noise = pipe.forward_diffusion(
            latents=attacked_image_latents,
            text_embeddings=pipe.get_text_embedding(""),
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        cosine_similarities, l1_similarities, l2_similarities = calculate_noise_similarities(reversed_noise, args.n_noise_samples, pipe)
        
        initial_cosine = cosine_similarity(reversed_noise, initial_noise).item()
        initial_l1 = l1_similarity(reversed_noise, initial_noise).item()
        initial_l2 = l2_similarity(reversed_noise, initial_noise).item()

        attacked_cosine = cosine_similarity(attacked_reversed_noise, initial_noise).item()
        attacked_l1 = l1_similarity(attacked_reversed_noise, initial_noise).item()
        attacked_l2 = l2_similarity(attacked_reversed_noise, initial_noise).item()

        all_cosine_similarities.extend(cosine_similarities)
        all_l1_similarities.extend(l1_similarities)
        all_l2_similarities.extend(l2_similarities)
        all_initial_cosine.append(initial_cosine)
        all_initial_l1.append(initial_l1)
        all_initial_l2.append(initial_l2)
        all_attacked_cosine.append(attacked_cosine)
        all_attacked_l1.append(attacked_l1)
        all_attacked_l2.append(attacked_l2)

    # Plot histograms
    plot_histograms(all_cosine_similarities, all_initial_cosine, all_attacked_cosine, "Cosine Similarities", "cosine_similarities.png")
    plot_histograms(all_l1_similarities, all_initial_l1, all_attacked_l1, "L1 Similarities", "l1_similarities.png")
    plot_histograms(all_l2_similarities, all_initial_l2, all_attacked_l2, "L2 Similarities", "l2_similarities.png")
    # Calculate statistics
    cosine_random_mean, cosine_random_std = np.mean(all_cosine_similarities), np.std(all_cosine_similarities)
    l1_random_mean, l1_random_std = np.mean(all_l1_similarities), np.std(all_l1_similarities)
    l2_random_mean, l2_random_std = np.mean(all_l2_similarities), np.std(all_l2_similarities)
    cosine_true_mean, cosine_true_std = np.mean(all_initial_cosine), np.std(all_initial_cosine)
    l1_true_mean, l1_true_std = np.mean(all_initial_l1), np.std(all_initial_l1)
    l2_true_mean, l2_true_std = np.mean(all_initial_l2), np.std(all_initial_l2)
    cosine_attacked_mean, cosine_attacked_std = np.mean(all_attacked_cosine), np.std(all_attacked_cosine)
    l1_attacked_mean, l1_attacked_std = np.mean(all_attacked_l1), np.std(all_attacked_l1)
    l2_attacked_mean, l2_attacked_std = np.mean(all_attacked_l2), np.std(all_attacked_l2)

    # Print results
    print("Overall Results:")
    print("Cosine Similarity:")
    print(f"  Random   - Mean: {cosine_random_mean:.4f}, Std: {cosine_random_std:.4f}")
    print(f"  True     - Mean: {cosine_true_mean:.4f}, Std: {cosine_true_std:.4f}")
    print(f"  Attacked - Mean: {cosine_attacked_mean:.4f}, Std: {cosine_attacked_std:.4f}")
    print("L1 Similarity:")
    print(f"  Random   - Mean: {l1_random_mean:.4f}, Std: {l1_random_std:.4f}")
    print(f"  True     - Mean: {l1_true_mean:.4f}, Std: {l1_true_std:.4f}")
    print(f"  Attacked - Mean: {l1_attacked_mean:.4f}, Std: {l1_attacked_std:.4f}")
    print("L2 Similarity:")
    print(f"  Random   - Mean: {l2_random_mean:.4f}, Std: {l2_random_std:.4f}")
    print(f"  True     - Mean: {l2_true_mean:.4f}, Std: {l2_true_std:.4f}")
    print(f"  Attacked - Mean: {l2_attacked_mean:.4f}, Std: {l2_attacked_std:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noise Comparison with Attack')
    parser.add_argument('--run_name', default='noise_comparison_with_attack')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true', default=True)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--n_noise_samples', default=10000, type=int)
    parser.add_argument('--attack_params', nargs='+', type=float, default=[0, 10]) 

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)