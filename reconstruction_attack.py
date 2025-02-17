import argparse
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from io_utils import *
from optim_utils import *
from PIL import Image
import os
from torchvision.transforms import ToPILImage

def cosine_similarity(a, b):
    return nn.functional.cosine_similarity(a.view(-1), b.view(-1), dim=0)

def l2_distance(a, b):
    return torch.norm(a - b)

def plot_histograms(results, title, filename):
    sns.set_style("whitegrid")
    colors = sns.color_palette("deep", len(results))

    metrics = ['cosine']
    approach_names = {
        'gen_rev_real': 'Generate -> Reverse (Real Model)',
        'gen_real_rev_open_gen_open_rev_real': 'Gen (Real) -> Rev (Open) -> Gen (Open) -> Rev (Real)'
    }

    os.makedirs("individual_histograms", exist_ok=True)

    for metric in metrics:
        for i, (approach, data) in enumerate(results.items()):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.histplot(data[metric], kde=True, stat="density", color=colors[i], ax=ax)
            
            mean_value = np.mean(data[metric])
            std_value = np.std(data[metric])
            
            ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.4f}')
            
            ax.set_title(f"{approach_names[approach]}\n{metric.upper()} Similarity", fontsize=16, fontweight='bold')
            ax.set_xlabel(f"{metric.capitalize()} Similarity", fontsize=16)
            ax.set_ylabel("Density", fontsize=16)
            ax.legend(fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            plt_filename = f"individual_histograms/{metric}_{approach}_{filename}"
            plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
            plt_filename = f"individual_histograms/{metric}_{approach}_{filename}"
            plt.savefig(plt_filename, dpi=300, bbox_inches='tight')

            wandb.log({
                f"histograms/{metric}/{approach}": wandb.Image(plt_filename),
                f"metrics/{metric}/{approach}/mean": mean_value,
                f"metrics/{metric}/{approach}/std": std_value
            })

            plt.close()

    # Create a summary plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    fig.suptitle(f"{title} - Summary", fontsize=20, fontweight='bold')

    for m, metric in enumerate(metrics):
        for i, (approach, data) in enumerate(results.items()):
            sns.kdeplot(data[metric], ax=axes[m], label=approach_names[approach], color=colors[i])
        
        axes[m].set_title(f"{metric.upper()} Similarity", fontsize=16, fontweight='bold')
        axes[m].set_xlabel(f"{metric.capitalize()} Similarity", fontsize=14)
        axes[m].set_ylabel("Density", fontsize=14)
        axes[m].legend(fontsize=10, loc='upper right')
        axes[m].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    summary_filename = f"individual_histograms/summary_{filename}"
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
    # if wandb.run is not None:
    #     wandb.log({"summary_histogram": wandb.Image(summary_filename)})
    plt.close()

def main(args):
    if args.with_tracking:
        wandb.init(project='Paper_Results', name='Noise_Comparison', tags=['noise_comparison'])
        wandb.config.update(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load your original model
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
    )
    pipe = pipe.to(device)

    # Load the open-source diffusion inverse model
    open_source_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16,
    )
    open_source_pipe = open_source_pipe.to(device)
    open_source_pipe.scheduler = DDIMScheduler.from_config(open_source_pipe.scheduler.config)
    open_source_pipe.scheduler.set_timesteps(args.num_inference_steps)

    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings_real = pipe.get_text_embedding(tester_prompt)
    text_embeddings_open = open_source_pipe.tokenizer(
        tester_prompt,
        padding="max_length",
        max_length=open_source_pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    text_embeddings_open = open_source_pipe.text_encoder(text_embeddings_open)[0]

    results = {
        'gen_rev_real': {'cosine': []},
        'gen_real_rev_open_gen_open_rev_real': {'cosine': []}
    }

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        original_prompt = dataset[i][prompt_key]
                    
        set_random_seed(seed)
        
        # Generate initial noise
        initial_noise = pipe.get_random_latents()
        
        # Generate image with real model
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

        # Approach 1: gen -> rev (all real model)
        reversed_noise_real = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings_real,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )
        results['gen_rev_real']['cosine'].append(cosine_similarity(reversed_noise_real, initial_noise).item())

        open_source_latents = open_source_pipe.vae.encode(img).latent_dist.sample()
        open_source_latents = open_source_latents * open_source_pipe.vae.config.scaling_factor
        reversed_noise_open = open_source_pipe.scheduler.add_noise(
            open_source_latents,
            torch.randn_like(open_source_latents),
            torch.tensor([args.test_num_inference_steps])
        )


        # Approach 2: gen (real) -> rev (open) -> gen (open) -> rev (real)
        with torch.no_grad():
            noisy_latents = reversed_noise_open
            for t in open_source_pipe.scheduler.timesteps:
                noise_pred = open_source_pipe.unet(noisy_latents, t, encoder_hidden_states=text_embeddings_open).sample
                noisy_latents = open_source_pipe.scheduler.step(noise_pred, t, noisy_latents).prev_sample
            generated_image_5 = open_source_pipe.vae.decode(noisy_latents / open_source_pipe.vae.config.scaling_factor).sample
        
        generated_image_5_pil = ToPILImage()(generated_image_5.squeeze().cpu())
        img_5 = transform_img(generated_image_5_pil).unsqueeze(0).to(noisy_latents.dtype).to(device)
        image_latents_5 = pipe.get_image_latents(img_5, sample=False)
        reversed_noise_real_5 = pipe.forward_diffusion(
            latents=image_latents_5,
            text_embeddings=text_embeddings_real,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )
        results['gen_real_rev_open_gen_open_rev_real']['cosine'].append(cosine_similarity(reversed_noise_real_5, reversed_noise_open).item())

    # Plot histograms and log results
    os.makedirs("histograms", exist_ok=True)
    plot_histograms(results, "Noise Comparison with Different Approaches", "noise_comparison.png")

    if args.with_tracking:
        table_data = []
        for approach, metrics in results.items():
            for metric, data in metrics.items():
                mean_value = np.mean(data)
                std_value = np.std(data)
                wandb.log({
                    f"{metric}_{approach}_mean": mean_value,
                    f"{metric}_{approach}_std": std_value,
                })
                table_data.append([approach, metric.upper(), mean_value, std_value])

        table = wandb.Table(data=table_data, columns=["Approach", "Metric", "Mean", "Std"])
        wandb.log({"results_summary": table})

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

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)