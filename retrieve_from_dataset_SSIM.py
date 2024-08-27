# %% 
import json
import os
import torch
from argparse import ArgumentParser
from ddim import DDIM, get_selection_schedule
from ddpm_torch import *
try:
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_DICT.keys(), default="cifar10")
    parser.add_argument("--config_path", type=str, default='./configs/cifar10.json')
    parser.add_argument("--config_dir", type=str, default='./configs')
    parser.add_argument("--chkpt_dir", type=str, default='./chkpt')
    parser.add_argument("--chkpt_path", type=str, default=None)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--skip-schedule", choices=["linear", "quadratic"], default="linear", type=str)
    parser.add_argument("--subseq-size", default=50, type=int)
    parser.add_argument("--eta", default=0., type=float)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--use_ema", action="store_true")
    
    args = parser.parse_args()
    config_path = args.config_path  
    dataset = args.dataset
    config_dir = config_path.rstrip("/")
    chkpt_path = args.chkpt_path
    chkpt_dir = chkpt_path.rstrip("/")
    use_ddim = args.use_ddim
    skip_schedule = args.skip_schedule
    eta = args.eta
    num_gpus = args.num_gpus
    device = args.device
    use_emas = args.use_ema
except:
    config_path = './configs/cifar10.json'
    dataset = 'cifar10'
    config_dir = './configs'    
    chkpt_path = '/home/hl-fury/mariarosaria.briglia/ddpm-torch/models/adv-ddpm/L2/adv-post-2024-08-08T160539047657/adv-post-2024-08-08T160539047657'
    chkpt_dir = chkpt_path.rstrip("/")
    use_ddim = False
    skip_schedule = 'linear'
    subseq_size = 50
    eta = 0
    num_gpus = 1    
    device = 'cuda' 
    use_ema = False
    

# %%
if config_path is None:
    config_path = os.path.join(config_dir, dataset + ".json")
with open(config_path, "r") as f:
    meta_config = json.load(f)
exp_name = os.path.basename(config_path)[:-5]

dataset = meta_config.get("dataset", dataset)
in_channels = DATASET_INFO[dataset]["channels"]
image_res = DATASET_INFO[dataset]["resolution"][0]
input_shape = (in_channels, image_res, image_res)

diffusion_kwargs = meta_config["diffusion"]
beta_schedule = diffusion_kwargs.pop("beta_schedule")
beta_start = diffusion_kwargs.pop("beta_start")
beta_end = diffusion_kwargs.pop("beta_end")
num_diffusion_timesteps = diffusion_kwargs.pop("timesteps")
betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)
# %%

if use_ddim:
    diffusion_kwargs["model_var_type"] = "fixed-small"
    skip_schedule = skip_schedule
    eta = eta
    subseq_size = subseq_size
    subsequence = get_selection_schedule(skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps)
    diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)
else:
    diffusion = GaussianDiffusion(betas, **diffusion_kwargs)
rank = 0
device = torch.device(f"cuda:{rank}" if num_gpus > 1 else device)
block_size = meta_config["model"].pop("block_size", 1)
model = UNet(out_channels=in_channels, **meta_config["model"])
if block_size > 1:
    pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
    post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
    model = ModelWrapper(model, pre_transform, post_transform)
model.to(device)
chkpt_dir = chkpt_dir
chkpt_path = chkpt_path+'_1040.pt' or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
folder_name = os.path.basename(chkpt_path)[:-3]  # truncated at file extension
use_ema = meta_config["train"].get("use_ema", use_ema)

state_dict = torch.load(chkpt_path, map_location=device)
try:
    if use_ema:
        state_dict = state_dict["ema"]["shadow"]
    else:
        state_dict = state_dict["model"]
    print("Loading checkpoint...", end=" ")
except KeyError:
    print("Not a valid checkpoint!")
    print("Try loading checkpoint directly as model weights...", end=" ")

for k in list(state_dict.keys()):
    if k.startswith("module."):  # state_dict of DDP
        state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)

try:
    model.load_state_dict(state_dict)
    del state_dict
    print("succeeded!")
except RuntimeError:
    model.load_state_dict(state_dict)
    print("failed!")
    exit(1)

model.eval()
for p in model.parameters():
    if p.requires_grad:
        p.requires_grad_(False)

# %%
# generate 10 images 
num_images = 10
shape = (num_images, ) + input_shape
x = diffusion.p_sample(model, shape=shape, device=device, noise=torch.randn(shape, device=device)).cpu()
images_to_be_evaluated = (x * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

# visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
for i in range(num_images):
    axes[i].imshow(images_to_be_evaluated[i])
    axes[i].axis("off")
plt.show()

 

# %%
dataset_test = DATASET_DICT[dataset]("/home/hl-fury/mariarosaria.briglia/ddpm-torch/data", "train")
# %%
test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=4)
# %%
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
k = 10
ssim_dict = {}
for index , image in enumerate(images_to_be_evaluated):
    ssims = []
    # No need to convert the image to grayscale
    for batch in test_dataloader:
        for img in batch:            
            img = img.numpy().transpose(1,2,0)
            # convert images to grayscale
            image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
            
            ssim_value =ssim(image_g, img_g, full=True)[0]
            ssims.append((ssim_value, img))
    
    ssims = sorted(ssims, key=lambda x: x[0], reverse=True)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    for i in range(5):
        print(f"SSIM: {ssims[i][0]}")
        image_ssim = ssims[i][1]
        # visualize a 3 channel image 
        axes[i].imshow(image_ssim)
        axes[i].axis("off")
    plt.show()
    ssim_dict[index] = (image,ssims[:k] )
# %%
for i in range(len(ssim_dict.keys())):
    # plot number of columns =1+len of ssim_dict[i][1]
    fig, axes = plt.subplots(1, 1+len(ssim_dict[i][1]), figsize=(20, 20))
    axes[0].imshow(ssim_dict[i][0])
    axes[0].axis("off")
    for j in range(len(ssim_dict[i][1])):
        axes[j+1].imshow(ssim_dict[i][1][j][1])
        axes[j+1].axis("off")
    plt.show()

# %%
# plot all images in a grid
fig, axes = plt.subplots(k, 11, figsize=(20, 20))
for i in range(k):
    axes[i][0].imshow(ssim_dict[i][0])
    axes[i][0].axis("off")
    for j in range(k):
        axes[i][j+1].imshow(ssim_dict[i][1][j][1])
        axes[i][j+1].axis("off")
plt.show()

# save the grid plot 
plt.savefig('grid_plot_1040.png')
# %%
if config_path is None:
    config_path = os.path.join(config_dir, dataset + ".json")
with open(config_path, "r") as f:
    meta_config = json.load(f)
exp_name = os.path.basename(config_path)[:-5]

dataset = meta_config.get("dataset", dataset)
in_channels = DATASET_INFO[dataset]["channels"]
image_res = DATASET_INFO[dataset]["resolution"][0]
input_shape = (in_channels, image_res, image_res)

diffusion_kwargs = meta_config["diffusion"]
beta_schedule = diffusion_kwargs.pop("beta_schedule")
beta_start = diffusion_kwargs.pop("beta_start")
beta_end = diffusion_kwargs.pop("beta_end")
num_diffusion_timesteps = diffusion_kwargs.pop("timesteps")
betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)
# %%

if use_ddim:
    diffusion_kwargs["model_var_type"] = "fixed-small"
    skip_schedule = skip_schedule
    eta = eta
    subseq_size = subseq_size
    subsequence = get_selection_schedule(skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps)
    diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)
else:
    diffusion = GaussianDiffusion(betas, **diffusion_kwargs)
rank = 0
device = torch.device(f"cuda:{rank}" if num_gpus > 1 else device)
block_size = meta_config["model"].pop("block_size", 1)
model = UNet(out_channels=in_channels, **meta_config["model"])
if block_size > 1:
    pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
    post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
    model = ModelWrapper(model, pre_transform, post_transform)
model.to(device)
chkpt_dir = chkpt_dir
chkpt_path = chkpt_path+'_480.pt' or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
folder_name = os.path.basename(chkpt_path)[:-3]  # truncated at file extension
use_ema = meta_config["train"].get("use_ema", use_ema)

state_dict = torch.load(chkpt_path, map_location=device)
try:
    if use_ema:
        state_dict = state_dict["ema"]["shadow"]
    else:
        state_dict = state_dict["model"]
    print("Loading checkpoint...", end=" ")
except KeyError:
    print("Not a valid checkpoint!")
    print("Try loading checkpoint directly as model weights...", end=" ")

for k in list(state_dict.keys()):
    if k.startswith("module."):  # state_dict of DDP
        state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)

try:
    model.load_state_dict(state_dict)
    del state_dict
    print("succeeded!")
except RuntimeError:
    model.load_state_dict(state_dict)
    print("failed!")
    exit(1)

model.eval()
for p in model.parameters():
    if p.requires_grad:
        p.requires_grad_(False)

# %%
# generate 10 images 
num_images = 10
shape = (num_images, ) + input_shape
x = diffusion.p_sample(model, shape=shape, device=device, noise=torch.randn(shape, device=device)).cpu()
images_to_be_evaluated = (x * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

# visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
for i in range(num_images):
    axes[i].imshow(images_to_be_evaluated[i])
    axes[i].axis("off")
plt.show()

 

# %%
dataset_test = DATASET_DICT[dataset]("/home/hl-fury/mariarosaria.briglia/ddpm-torch/data", "train")
# %%
test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=4)
# %%
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
k = 10
ssim_dict = {}
for index , image in enumerate(images_to_be_evaluated):
    ssims = []
    # No need to convert the image to grayscale
    for batch in test_dataloader:
        for img in batch:            
            img = img.numpy().transpose(1,2,0)
            # convert images to grayscale
            image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
            
            ssim_value =ssim(image_g, img_g, full=True)[0]
            ssims.append((ssim_value, img))
    
    ssims = sorted(ssims, key=lambda x: x[0], reverse=True)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    for i in range(5):
        print(f"SSIM: {ssims[i][0]}")
        image_ssim = ssims[i][1]
        # visualize a 3 channel image 
        axes[i].imshow(image_ssim)
        axes[i].axis("off")
    plt.show()
    ssim_dict[index] = (image,ssims[:k] )
# %%
for i in range(len(ssim_dict.keys())):
    # plot number of columns =1+len of ssim_dict[i][1]
    fig, axes = plt.subplots(1, 1+len(ssim_dict[i][1]), figsize=(20, 20))
    axes[0].imshow(ssim_dict[i][0])
    axes[0].axis("off")
    for j in range(len(ssim_dict[i][1])):
        axes[j+1].imshow(ssim_dict[i][1][j][1])
        axes[j+1].axis("off")
    plt.show()

# %%
# plot all images in a grid
fig, axes = plt.subplots(k, 11, figsize=(20, 20))
for i in range(k):
    axes[i][0].imshow(ssim_dict[i][0])
    axes[i][0].axis("off")
    for j in range(k):
        axes[i][j+1].imshow(ssim_dict[i][1][j][1])
        axes[i][j+1].axis("off")
plt.show()

# save the grid plot 
plt.savefig('grid_plot_480.png')
# %%
if config_path is None:
    config_path = os.path.join(config_dir, dataset + ".json")
with open(config_path, "r") as f:
    meta_config = json.load(f)
exp_name = os.path.basename(config_path)[:-5]

dataset = meta_config.get("dataset", dataset)
in_channels = DATASET_INFO[dataset]["channels"]
image_res = DATASET_INFO[dataset]["resolution"][0]
input_shape = (in_channels, image_res, image_res)

diffusion_kwargs = meta_config["diffusion"]
beta_schedule = diffusion_kwargs.pop("beta_schedule")
beta_start = diffusion_kwargs.pop("beta_start")
beta_end = diffusion_kwargs.pop("beta_end")
num_diffusion_timesteps = diffusion_kwargs.pop("timesteps")
betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)
# %%

if use_ddim:
    diffusion_kwargs["model_var_type"] = "fixed-small"
    skip_schedule = skip_schedule
    eta = eta
    subseq_size = subseq_size
    subsequence = get_selection_schedule(skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps)
    diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)
else:
    diffusion = GaussianDiffusion(betas, **diffusion_kwargs)
rank = 0
device = torch.device(f"cuda:{rank}" if num_gpus > 1 else device)
block_size = meta_config["model"].pop("block_size", 1)
model = UNet(out_channels=in_channels, **meta_config["model"])
if block_size > 1:
    pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
    post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
    model = ModelWrapper(model, pre_transform, post_transform)
model.to(device)
chkpt_dir = chkpt_dir
chkpt_path = chkpt_path +'_2040.pt' or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
folder_name = os.path.basename(chkpt_path)[:-3]  # truncated at file extension
use_ema = meta_config["train"].get("use_ema", use_ema)

state_dict = torch.load(chkpt_path, map_location=device)
try:
    if use_ema:
        state_dict = state_dict["ema"]["shadow"]
    else:
        state_dict = state_dict["model"]
    print("Loading checkpoint...", end=" ")
except KeyError:
    print("Not a valid checkpoint!")
    print("Try loading checkpoint directly as model weights...", end=" ")

for k in list(state_dict.keys()):
    if k.startswith("module."):  # state_dict of DDP
        state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)

try:
    model.load_state_dict(state_dict)
    del state_dict
    print("succeeded!")
except RuntimeError:
    model.load_state_dict(state_dict)
    print("failed!")
    exit(1)

model.eval()
for p in model.parameters():
    if p.requires_grad:
        p.requires_grad_(False)

# %%
# generate 10 images 
num_images = 10
shape = (num_images, ) + input_shape
x = diffusion.p_sample(model, shape=shape, device=device, noise=torch.randn(shape, device=device)).cpu()
images_to_be_evaluated = (x * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

# visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
for i in range(num_images):
    axes[i].imshow(images_to_be_evaluated[i])
    axes[i].axis("off")
plt.show()

 

# %%
dataset_test = DATASET_DICT[dataset]("/home/hl-fury/mariarosaria.briglia/ddpm-torch/data", "train")
# %%
test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=4)
# %%
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
k = 10
ssim_dict = {}
for index , image in enumerate(images_to_be_evaluated):
    ssims = []
    # No need to convert the image to grayscale
    for batch in test_dataloader:
        for img in batch:            
            img = img.numpy().transpose(1,2,0)
            # convert images to grayscale
            image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
            
            ssim_value =ssim(image_g, img_g, full=True)[0]
            ssims.append((ssim_value, img))
    
    ssims = sorted(ssims, key=lambda x: x[0], reverse=True)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    for i in range(5):
        print(f"SSIM: {ssims[i][0]}")
        image_ssim = ssims[i][1]
        # visualize a 3 channel image 
        axes[i].imshow(image_ssim)
        axes[i].axis("off")
    plt.show()
    ssim_dict[index] = (image,ssims[:k] )
# %%
for i in range(len(ssim_dict.keys())):
    # plot number of columns =1+len of ssim_dict[i][1]
    fig, axes = plt.subplots(1, 1+len(ssim_dict[i][1]), figsize=(20, 20))
    axes[0].imshow(ssim_dict[i][0])
    axes[0].axis("off")
    for j in range(len(ssim_dict[i][1])):
        axes[j+1].imshow(ssim_dict[i][1][j][1])
        axes[j+1].axis("off")
    plt.show()

# %%
# plot all images in a grid
fig, axes = plt.subplots(k, 11, figsize=(20, 20))
for i in range(k):
    axes[i][0].imshow(ssim_dict[i][0])
    axes[i][0].axis("off")
    for j in range(k):
        axes[i][j+1].imshow(ssim_dict[i][1][j][1])
        axes[i][j+1].axis("off")
plt.show()

# save the grid plot 
plt.savefig('grid_plot_2040.png')
# %%
