# %%
import json
import os
import torch
from argparse import ArgumentParser
from ddim import DDIM, get_selection_schedule
from ddpm_torch import *

# %%
save_path = "./classifier-plots"
if not os.path.exists(save_path):
    os.makedirs(save_path)

try:
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_DICT.keys(), default="cifar10")
    parser.add_argument("--config_path", type=str, default="./configs/cifar10.json")
    parser.add_argument("--config_dir", type=str, default="./configs")
    parser.add_argument("--chkpt_dir", type=str, default="./chkpt")
    parser.add_argument("--chkpt_path", type=str, default=None)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument(
        "--skip-schedule", choices=["linear", "quadratic"], default="linear", type=str
    )
    parser.add_argument("--subseq-size", default=50, type=int)
    parser.add_argument("--eta", default=0.0, type=float)
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
    use_ema = args.use_ema
except:
    config_path = "./configs/cifar10.json"
    dataset = "cifar10"
    config_dir = "./configs"
    chkpt_path = "/home/hl-fury/mariarosaria.briglia/ddpm-torch/models/adv-ddpm/L2/UNet/cifar10/cifar10_480.pt"
    chkpt_dir = chkpt_path.rstrip("/")
    use_ddim = False
    skip_schedule = "linear"
    subseq_size = 50
    eta = 0
    num_gpus = 1
    device = "cuda"
    use_ema = False

print('chkpt_path', chkpt_path) 

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
    subsequence = get_selection_schedule(
        skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps
    )
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
chkpt_path = chkpt_path or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
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
shape = (num_images,) + input_shape
x = diffusion.p_sample(
    model, shape=shape, device=device, noise=torch.randn(shape, device=device)
).cpu()
images_to_be_evaluated = (
    (x * 127.5 + 127.5)
    .round()
    .clamp(0, 255)
    .to(torch.uint8)
    .permute(0, 2, 3, 1)
    .numpy()
)
#  %%
# visualize
import matplotlib.pyplot as plt

plt.close()
fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
for i in range(num_images):
    axes[i].imshow(images_to_be_evaluated[i])
    axes[i].axis("off")
# %%
plt.savefig(
    os.path.join(
        save_path, f"generated_images-{''.join(chkpt_path.split('/')[-3:])}.png"
    )
)
plt.close()


# %%
dataset_test = DATASET_DICT[dataset](
    "/home/hl-fury/mariarosaria.briglia/ddpm-torch/data", "test"
)
dataset_train = DATASET_DICT[dataset](
    "/home/hl-fury/mariarosaria.briglia/ddpm-torch/data", "train"
)
# %%
test_dataloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=64, shuffle=True, num_workers=4
)
train_dataloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=64, shuffle=True, num_workers=4
)
# %%
import resnets

wr = resnets.wide_resnet28_10().to(device)
state_dict = torch.load(
    "/home/hl-fury/mariarosaria.briglia/ddpm-torch/model_best.pth.tar"
)["state_dict"]
# remove some of the name
for key in list(state_dict.keys()):
    if "module.models.0." in key:
        state_dict[key.replace("module.models.0.", "")] = state_dict.pop(key)
wr.load_state_dict(state_dict)
wr.eval()

print("Classifier loaded")

# %%

# instanciate the histogram of the classes
cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# %%
hist = {c: 0 for c in cifar10_classes}
# generate 10k imgaes and evaluate them with the classifier and populate the histogram
for i, images in enumerate(test_dataloader):
    # normalize the images in [-1,1] from [0, 255]
    images = (images - 127.5) / 127.5
    outputs = wr(images.to(device))
    class_predictions = torch.argmax(outputs, dim=1)
    for p in class_predictions:
        hist[cifar10_classes[p]] += 1


# plot the histogram
import matplotlib.pyplot as plt

plt.bar(hist.keys(), hist.values())
plt.savefig(os.path.join(save_path, f"histogram-training-set.png"))

plt.close()
# %%
# generate 10k images and evaluate them with the classifier and populate the histogram of the classified samples according to the classifier

from tqdm import tqdm

bs = 100
num_batches = 100
shape = (bs,) + input_shape
generated_img_histogram = {c: 0 for c in cifar10_classes}
for i in tqdm(range(num_batches)):
    x = diffusion.p_sample(
        model, shape=shape, device=device, noise=torch.randn(shape, device=device)
    ).cpu()
    images_to_be_evaluated = (
        (x * 127.5 + 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    img = x.float().to(device)
    outputs = wr(img)
    class_predictions = torch.argmax(outputs, dim=1)
    for cp in class_predictions:
        generated_img_histogram[cifar10_classes[cp]] += 1
print(generated_img_histogram)

# suplot the histogram on the same as the previous one

plt.bar(hist.keys(), hist.values())
plt.bar(generated_img_histogram.keys(), generated_img_histogram.values(), alpha=0.5)
plt.savefig(
    os.path.join(
        save_path, f"histogram-generated-{''.join(chkpt_path.split('/')[-3:])}.png"
    )
)
plt.close()
# %%
# plot all the images 100 of a batch in a grid 10 x 10
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(images_to_be_evaluated[i * 10 + j])
        axes[i, j].axis("off")
plt.savefig(
    os.path.join(
        save_path, f"generated_images-{''.join(chkpt_path.split('/')[-3:])}.png"
    )
)
plt.close()
# %%

# I get the 100 images batch from the test dataloader
batch = next(iter(test_dataloader))
# generate 100 images
shape = (batch.shape[0],) + input_shape
x = diffusion.p_sample(
    model, shape=shape, device=device, noise=torch.randn(shape, device=device)
).cpu()
images_to_be_evaluated = (
    (x * 127.5 + 127.5)
    .round()
    .clamp(0, 255)
    .to(torch.uint8)
    .permute(0, 2, 3, 1)
    .numpy()
)
# %%
# plot all the images in a grid 10 x 10
import matplotlib.pyplot as plt

fig, axes = plt.subplots(8, 8, figsize=(20, 20))
for i in range(8):
    for j in range(8):
        axes[i, j].imshow(images_to_be_evaluated[i * 8 + j])
        axes[i, j].axis("off")
plt.show()
plt.close()


# %%
def extract_features(model, images, layer_name):
    """
    Extracts features from a specified layer of the model for a batch of images.

    Parameters:
        model (torch.nn.Module): The pre-trained model.
        images (torch.Tensor): A batch of images.
        layer_name (str): The name of the layer from which to extract features.

    Returns:
        torch.Tensor: The extracted features.
    """
    features = None

    def hook(module, input, output):
        nonlocal features
        features = output

    # Attach the hook to the specified layer
    layer = dict([*model.named_modules()])[layer_name]
    handle = layer.register_forward_hook(hook)

    # Pass the images through the model
    with torch.no_grad():
        model(images)

    # Remove the hook
    handle.remove()

    return features


# Get all module names
module_names = [name for name, _ in wr.named_modules()]

# The penultimate layer will be the second last item in this list
penultimate_layer_name = module_names[-4]

print(f"The penultimate layer name is: {penultimate_layer_name}")

import torch

# Assuming `images` is a numpy array, convert it to a PyTorch Tensor
images_tensor = x

# Ensure that the tensor is on the same device as the model (e.g., CPU or GPU)
images_tensor = images_tensor.to(device)

# Extract features from the penultimate layer
features = extract_features(wr, images_tensor, penultimate_layer_name)

# %%

# I want to evaluate the most similar samples of the dataset to the generated images through the penultimate layer of the classifier

from tqdm import tqdm
import matplotlib.pyplot as plt

x = images_to_be_evaluated[:10]
print(x.shape)
# %%
images_ds_all = {}
for i, im in tqdm(enumerate(x)):
    im = (
        torch.from_numpy((im - 127.5) / 127.5)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        .float()
    )
    print("min im", im.min(), "max im", im.max())
    print(im.shape)

    dists = []
    dist_dict = {}
    images_ds = []
    emb_im = extract_features(wr, im, penultimate_layer_name)
    for batch in test_dataloader:
        for image in batch:
            image_b = image.unsqueeze(0).to(device).float()
            # cast image in [-1,1]
            image_b = (image_b - 127.5) / 127.5
            emb = extract_features(wr, image_b, penultimate_layer_name)
            # Compute pairwise distance and take the mean
            distance = torch.nn.functional.pairwise_distance(emb_im, emb).mean()
            # dists.append(distance.item())
            # images_ds.append(image)
            dist_dict[image] = distance.item()

    for k, v in sorted(dist_dict.items(), key=lambda item: item[1])[:10]:
        images_ds.append(k)
        dists.append(v)
    images_ds_all[i] = images_ds
# %%
plt.close()
fig, axes = plt.subplots(len(images_ds_all.keys()), 11, figsize=(20, 20))
for i, im in tqdm(enumerate(images_tensor[:10])):
    # normalize the images from [-1,1] in [0, 255]
    im = (im * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    axes[i, 0].imshow(im.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[i, 0].axis("off")
    for j, img in enumerate(images_ds_all[i]):
        axes[i, j + 1].imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axes[i, j + 1].axis("off")
plt.savefig(
    os.path.join(
        save_path,
        f"most-similar-{''.join(chkpt_path.split('/')[-3:])}-{i}.png",
    )
)
plt.close()


# %%
