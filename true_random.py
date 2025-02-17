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

def plot_histograms(random_data, true_data, title, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Set the style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", 2)
    
    # Plot for random noises
    sns.histplot(random_data, kde=True, stat="density", color=colors[0], ax=ax1)
    ax1.axvline(np.mean(random_data), color='red', linestyle='dashed', linewidth=2, label='Mean')
    ax1.set_title(f"Random Noise: {title}", fontsize=16, fontweight='bold')
    ax1.set_xlabel('Similarity', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.text(np.mean(random_data), ax1.get_ylim()[1], f'Mean: {np.mean(random_data):.4f}', 
             horizontalalignment='center', verticalalignment='bottom', color='red', fontweight='bold')

    # Plot for true noises
    sns.histplot(true_data, kde=True, stat="density", color=colors[1], ax=ax2)
    ax2.axvline(np.mean(true_data), color='red', linestyle='dashed', linewidth=2, label='Mean')
    ax2.set_title(f"True Noise: {title}", fontsize=16, fontweight='bold')
    ax2.set_xlabel('Similarity', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.text(np.mean(true_data), ax2.get_ylim()[1], f'Mean: {np.mean(true_data):.4f}', 
             horizontalalignment='center', verticalalignment='bottom', color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    wandb.init(project='Paper_Results', name='Noise_Match', tags=['noise_match'])
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

        img = transform_img(generated_image).unsqueeze(0).to(initial_noise.dtype).to(device)
        image_latents = pipe.get_image_latents(img, sample=False)
        reversed_noise = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=pipe.get_text_embedding(""),
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        cosine_similarities, l1_similarities, l2_similarities = calculate_noise_similarities(reversed_noise, args.n_noise_samples, pipe)
        
        initial_cosine = cosine_similarity(reversed_noise, initial_noise).item()
        initial_l1 = l1_similarity(reversed_noise, initial_noise).item()
        initial_l2 = l2_similarity(reversed_noise, initial_noise).item()

        all_cosine_similarities.extend(cosine_similarities)
        all_l1_similarities.extend(l1_similarities)
        all_l2_similarities.extend(l2_similarities)
        all_initial_cosine.append(initial_cosine)
        all_initial_l1.append(initial_l1)
        all_initial_l2.append(initial_l2)

    # Plot histograms
    plot_histograms(all_cosine_similarities, all_initial_cosine, "Cosine Similarities", "cosine_similarities.png")
    plot_histograms(all_l1_similarities, all_initial_l1, "L1 Similarities", "l1_similarities.png")
    plot_histograms(all_l2_similarities, all_initial_l2, "L2 Similarities", "l2_similarities.png")

    # Calculate statistics
    cosine_random_mean, cosine_random_std = np.mean(all_cosine_similarities), np.std(all_cosine_similarities)
    l1_random_mean, l1_random_std = np.mean(all_l1_similarities), np.std(all_l1_similarities)
    l2_random_mean, l2_random_std = np.mean(all_l2_similarities), np.std(all_l2_similarities)
    cosine_true_mean, cosine_true_std = np.mean(all_initial_cosine), np.std(all_initial_cosine)
    l1_true_mean, l1_true_std = np.mean(all_initial_l1), np.std(all_initial_l1)
    l2_true_mean, l2_true_std = np.mean(all_initial_l2), np.std(all_initial_l2)

    # Log results
    if args.with_tracking:
        wandb.log({
            "cosine_similarities": wandb.Image("cosine_similarities.png"),
            "l1_similarities": wandb.Image("l1_similarities.png"),
            "l2_similarities": wandb.Image("l2_similarities.png"),
            "cosine_random_mean": cosine_random_mean,
            "cosine_random_std": cosine_random_std,
            "cosine_true_mean": cosine_true_mean,
            "cosine_true_std": cosine_true_std,
            "l1_random_mean": l1_random_mean,
            "l1_random_std": l1_random_std,
            "l1_true_mean": l1_true_mean,
            "l1_true_std": l1_true_std,
            "l2_random_mean": l2_random_mean,
            "l2_random_std": l2_random_std,
            "l2_true_mean": l2_true_mean,
            "l2_true_std": l2_true_std,
        })

    # Print results
    print("Overall Results:")
    print("Cosine Similarity:")
    print(f"  Random - Mean: {cosine_random_mean:.4f}, Std: {cosine_random_std:.4f}")
    print(f"  True   - Mean: {cosine_true_mean:.4f}, Std: {cosine_true_std:.4f}")
    print("L1 Similarity:")
    print(f"  Random - Mean: {l1_random_mean:.4f}, Std: {l1_random_std:.4f}")
    print(f"  True   - Mean: {l1_true_mean:.4f}, Std: {l1_true_std:.4f}")
    print("L2 Similarity:")
    print(f"  Random - Mean: {l2_random_mean:.4f}, Std: {l2_random_std:.4f}")
    print(f"  True   - Mean: {l2_true_mean:.4f}, Std: {l2_true_std:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noise Comparison')
    parser.add_argument('--run_name', default='noise_comparison')
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

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)