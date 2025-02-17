import torch
from torchvision import transforms
from datasets import load_dataset
import os
from PIL import Image, ImageFilter, ImageDraw
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import math
from wmattacker import DiffWMAttacker, VAEWMAttacker, JPEGAttacker
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline
from inverse_stable_diffusion import InversableStableDiffusionPipeline
import copy
import open_clip
from att_src.diffusers.pipelines.stable_diffusion.pipeline_re_sd import ReSDPipeline
from torch import nn
from tqdm import tqdm


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img2)
        
    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key


# Removed circle_mask function since we no longer need to create circular masks

def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool, device=device)

    # Removed the circular mask condition
    # Instead, we will rely solely on the LLM mask in the watermark injection process
    if args.w_mask_shape == 'llm':
        # Placeholder: The actual LLM mask will be integrated elsewhere
        pass
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def get_watermarking_pattern(pipe, args, device, shape=None):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    # Removed all circular mask related watermark pattern generation
    # Now, watermark patterns are generated based on LLM-suggested coordinates directly

    if 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    else:
        raise NotImplementedError(f'w_pattern: {args.w_pattern} not implemented')

    return gt_patch


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    # Removed FFT-based watermark injection since we're no longer using circular masks
    # Now, inject watermark directly based on the LLM mask

    if args.w_injection == 'complex':
        # If complex injection is still required, ensure it's based on LLM mask only
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        raise NotImplementedError(f'w_injection: {args.w_injection}')

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement or 'seed' in args.w_measurement:
        reversed_latents_no_w_eval = reversed_latents_no_w
        reversed_latents_w_eval = reversed_latents_w
        target_patch = gt_patch
    else:
        raise NotImplementedError(f'w_measurement: {args.w_measurement}')

    # Ensure watermarking_mask is boolean
    watermarking_mask = watermarking_mask.bool()

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_eval[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_eval[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        raise NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric


def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w


def evaluate_pixel_similarity(initial_noise, reversed_noise, threshold=0.1):
    difference = torch.abs(initial_noise - reversed_noise)
    mean_difference = torch.mean(difference)
    return mean_difference

def evaluate_fourier_similarity(initial_noise, reversed_noise, threshold=0.1):
    initial_fft = torch.fft.fft2(initial_noise)
    reversed_fft = torch.fft.fft2(reversed_noise)
    difference = torch.abs(initial_fft - reversed_fft)
    mean_difference = torch.mean(difference)
    return mean_difference


def evaluate_pixel_cosine_similarity(initial_noise, reversed_noise):
    return F.cosine_similarity(initial_noise.flatten(), reversed_noise.flatten(), dim=0).item()

def evaluate_fourier_cosine_similarity(initial_noise, reversed_noise):
    initial_fft = torch.fft.fft2(initial_noise)
    reversed_fft = torch.fft.fft2(reversed_noise)
    initial_fft = initial_fft.float()
    reversed_fft = reversed_fft.float()
    return F.cosine_similarity(initial_fft.flatten(), reversed_fft.flatten(), dim=0).item()

def remove_nan_values(preds, labels):
    """Remove NaN values from predictions and corresponding labels."""
    valid_indices = ~np.isnan(preds)
    return preds[valid_indices], np.array(labels)[valid_indices]


ctr = 0


def get_llm_mask(image_path, image_size):
    global ctr
    org_img_path, org_img_size = image_path, image_size
    image_path = os.path.abspath(image_path)
    
    prompt = f"""
    Analyze this image and identify areas that could be slightly modified without altering the overall meaning or key elements. Provide the coordinates (x1, y1, x2, y2) for these areas. Ensure that the suggested areas are sufficiently large and that the coordinates are spread across different regions of the image. At least one coordinate should be on the main object of the image. Avoid suggesting coordinates that are close to each other. Format your response as a JSON object with a 'coordinates' key containing a list of coordinate tuples.

    For example:
    {{
        "coordinates": [[10, 20, 50, 60], [100, 150, 200, 250]]
    }}

    Here's the image:
    {image_path}"""

    # Use Ollama to interact with the llava model
    response = ollama.chat(model='llava-llama3', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    result = response['message']['content']

    try:
        # Try to extract JSON from the response
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = result[json_start:json_end]
            data = json.loads(json_str)
            coords = data.get('coordinates', [])
        else:
            if ctr < 20:
                ctr += 1
                return get_llm_mask(org_img_path, org_img_size)
            else:
                raise ValueError("No valid JSON found in the response.")
    except json.JSONDecodeError:
        if ctr < 20:
            ctr += 1
            return get_llm_mask(org_img_path, org_img_size)
        else:
            raise ValueError("Error parsing JSON from LLaVA response.")
    
    # Create mask based on the coordinates
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Calculate minimum size (5% of image dimensions)
    min_width = int(image_size[0] * 0.1)
    min_height = int(image_size[1] * 0.1)
    
    for coord in coords:
        try:
            x1, y1, x2, y2 = coord
            
            # Ensure x2 > x1 and y2 > y1
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, image_size[0] - min_width))
            y1 = max(0, min(y1, image_size[1] - min_height))
            x2 = min(image_size[0], max(x2, x1 + min_width))
            y2 = min(image_size[1], max(y2, y1 + min_height))
            
            # Ensure the area is at least 5% of the image size
            width = x2 - x1
            height = y2 - y1
            if width < min_width:
                x2 = min(image_size[0], x1 + min_width)
            if height < min_height:
                y2 = min(image_size[1], y1 + min_height)
            
            # Draw rectangle on the mask
            draw.rectangle([x1, y1, x2, y2], fill=255)
            
        except (ValueError, TypeError) as e:
            print(f"Invalid coordinate: {coord}. Skipping. Error: {e}")
    
    return ToTensor()(mask)


def get_spatial_watermarking_pattern(pipe, args, device, dtype=torch.float32, shape=None):
    set_random_seed(args.w_seed)
    if shape is not None:
        noise = torch.randn(*shape, device=device, dtype=dtype)
    else:
        noise = pipe.get_random_latents().to(dtype)

    # Removed the creation of a circular mask
    # Watermark is generated based on LLM mask only

    # if 'seed_zeros' in args.w_pattern:
    #     watermark = torch.zeros_like(noise)
    # elif 'seed_rand' in args.w_pattern:
    #     watermark = noise.clone()
    # elif 'const' in args.w_pattern:
    watermark = torch.full_like(noise, args.w_pattern_const)
    # else:
    #     raise NotImplementedError(f'w_pattern: {args.w_pattern} not implemented')

    return watermark


def integrate_llm_mask(init_latents_w, image_path, image_size, args, device):
    llm_mask = get_llm_mask(image_path, image_size)
    llm_mask = llm_mask.to(device)
    
    # Resize the mask to match the latent space dimensions
    latent_size = init_latents_w.shape[-1]
    llm_mask = torch.nn.functional.interpolate(llm_mask.unsqueeze(0), size=(latent_size, latent_size), mode='bilinear', align_corners=False)
    llm_mask = llm_mask.squeeze(0)
    
    # Convert the mask to boolean
    llm_mask = llm_mask > 0.5
    
    # Expand dimensions to match init_latents_w
    llm_mask = llm_mask.expand_as(init_latents_w)
    
    return llm_mask


def inject_watermark_to_noise(noise, watermark, llm_mask):
    # Ensure all inputs are on the same device and dtype
    device = noise.device
    dtype = noise.dtype
    noise = noise.to(device).to(dtype)
    watermark = watermark.to(device).to(dtype)
    llm_mask = llm_mask.to(device).to(dtype)

    # Ensure llm_mask is binary
    llm_mask = (llm_mask > 0.5).float()

    # Removed the creation and use of a circular mask
    # Now, the watermark is applied directly based on the LLM mask

    # Apply the watermark only to the areas specified by the LLM mask
    watermarked_noise = noise * (1 - llm_mask) + watermark * llm_mask

    return watermarked_noise


def apply_watermark_to_inpainting_noise(pipe, noise, temp_image_path, args, device):
    # Determine the dtype of the input noise
    dtype = noise.dtype
    
    # Generate the watermark pattern with the same dtype as the noise
    watermark = get_spatial_watermarking_pattern(pipe, args, device, dtype=dtype, shape=noise.shape)
    
    # Get the LLM-suggested mask
    llm_mask = integrate_llm_mask(noise, temp_image_path, (args.image_length, args.image_length), args, device)
    
    # Ensure llm_mask is the same dtype as noise
    llm_mask = llm_mask.to(dtype)
    
    # Apply the watermark to the noise using the LLM-suggested areas
    watermarked_noise = inject_watermark_to_noise(noise, watermark, llm_mask)
    
    return watermarked_noise, llm_mask


# Debugging function
def print_tensor_info(tensor, name):
    print(f"{name} - Shape: {tensor.shape}, Type: {tensor.dtype}, Device: {tensor.device}")
    print(f"{name} - Min: {tensor.min().item()}, Max: {tensor.max().item()}")


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_ssim(a, b):
    return ssim(a, b, data_range=1.).item()


def eval_psnr_ssim_msssim(ori_img_array, new_img_array):
    # Convert numpy arrays to tensors
    ori_x = torch.from_numpy(ori_img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    new_x = torch.from_numpy(new_img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x), compute_msssim(ori_x, new_x)


def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def save_latent_noise_as_image(latents, filename):
    # Convert latents to CPU and detach from computation graph
    latents = latents.cpu().detach()

    # Take the mean across the channel dimension
    mean_latents = latents.mean(dim=1)

    # Normalize to [0, 1] range
    normalized = (mean_latents - mean_latents.min()) / (mean_latents.max() - mean_latents.min())

    # Convert to numpy array and then to PIL Image
    image_array = (normalized.numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image_array.squeeze(), mode='L')
    
    # Save the image
    image.save(filename)


# def apply_attack(image, attack_type, attack_params, pipe):
#     if attack_type == 'gaussian_noise':
#         mean, std = attack_params
#         noise = np.random.normal(mean, std, image.size)
#         noisy_image = np.array(image) + noise
#         noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
#         return Image.fromarray(noisy_image)
#     elif attack_type == 'jpeg_compression':
#         quality = attack_params
#         image.save('temp.jpg', 'JPEG', quality=quality)
#         return Image.open('temp.jpg')
#     elif attack_type == 'diffusion':
#         attacker = DiffWMAttacker()
#         attacked_image = attacker.attack(image)
#         return attacked_image
#     elif attack_type == 'vae':
#         attacker = VAEWMAttacker()
#         attacked_image = attacker.attack(image)
#         return attacked_image
#     elif attack_type == 'diff_attacker_60':
#         pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
#         pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
#         attacker = DiffWMAttacker(pipe, batch_size=5, noise_step=60, captions={})
#         # Save the image to a temporary file
#         temp_file = 'temp_attack_image.png'
#         image.save(temp_file)
#         attacked_image, attacked_latents = attacker.attack([temp_file], ['attacked_image.png'])
#         # Remove the temporary file
#         os.remove(temp_file)
#         return Image.open('attacked_image.png'), attacked_latents
#     else:
#         raise NotImplementedError(f'Attack type {attack_type} not implemented')
    

def evaluate_attack(wmarker, attackers, ori_img_paths, output_path):
    detect_att_results = {}
    print('*' * 50)
    print(f'Watermark: {wmarker}')
    detect_att_results[wmarker] = {}
    for attacker_name, attacker in attackers.items():
        print(f'Attacker: {attacker_name}')
        bit_accs = []
        wm_successes = []
        for ori_img_path in ori_img_paths:
            img_name = os.path.basename(ori_img_path)
            att_img_path = os.path.join(output_path, wmarker, attacker_name, img_name)
            att_text = wmarker.decode(att_img_path)
            try:
                if type(att_text) == bytes:
                    a = bytearray_to_bits('test'.encode('utf-8'))
                    b = bytearray_to_bits(att_text)
                elif type(att_text) == str:
                    a = bytearray_to_bits('test'.encode('utf-8'))
                    b = bytearray_to_bits(att_text.encode('utf-8'))
                bit_acc = (np.array(a) ==  np.array(b)).mean()
                bit_accs.append(bit_acc)
                if bit_acc > 24/32:
                    wm_successes.append(img_name)
            except Exception as e:
                print('#' * 50)
                print(f'failed to decode {att_text}', type(att_text), len(att_text), e)
                pass
        detect_att_results[wmarker][attacker_name] = {}
        detect_att_results[wmarker][attacker_name]['bit_acc'] = np.array(bit_accs).mean()
        detect_att_results[wmarker][attacker_name]['wm_success'] = len(wm_successes) / len(ori_img_paths)
    return detect_att_results

def latents_to_image(latents):
    # Convert latents to CPU and detach from computation graph
    latents = latents.cpu().detach()

    # Take the mean across the channel dimension
    mean_latents = latents.mean(dim=1)

    # Normalize to [0, 1] range
    normalized = (mean_latents - mean_latents.min()) / (mean_latents.max() - mean_latents.min())

    # Convert to numpy array and then to PIL Image
    image_array = (normalized.numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image_array.squeeze(), mode='L')
    
    # Save the image
    return image




att_i = 0 

def apply_attack(orig_image,attacker_name, attack_params):

    if attacker_name == 'diff_attacker_60':
        global att_i
        temp_file = f'temp_attack_image_{att_i}.png'
        att_i += 1
        pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
        pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
        attacker = DiffWMAttacker(pipe, batch_size=5, noise_step=60, captions={})
        orig_image.save(temp_file)
        attacked_image, attacked_latents = attacker.attack([temp_file], ['attacked_image.png'])
        os.remove(temp_file)
    else:
        raise ValueError(f"Unknown attacker: {attacker_name}")
    
    return attacked_image, attacked_latents
    