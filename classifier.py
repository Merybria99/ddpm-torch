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
    use_emas = args.use_ema
except:
    config_path = "./configs/cifar10.json"
    dataset = "cifar10"
    config_dir = "./configs"
    chkpt_path = "/home/hl-fury/mariarosaria.briglia/ddpm-torch/models/adv-ddpm/L2/adv-post-2024-08-08T160539047657/adv-post-2024-08-08T160539047657"
    chkpt_dir = chkpt_path.rstrip("/")
    use_ddim = False
    skip_schedule = "linear"
    subseq_size = 50
    eta = 0
    num_gpus = 1
    device = "cuda"
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
chkpt_path = chkpt_path + "_1040.pt" or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
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

# visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
for i in range(num_images):
    axes[i].imshow(images_to_be_evaluated[i])
    axes[i].axis("off")
plt.show()


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
    # plot the image of the batch in a grid 8 x 8
    # plt.figure(figsize=(20, 20))
    # for i in range(8):
    #     for j in range(8):
    #         plt.subplot(8, 8, i*8+j+1)
    #         plt.imshow(images[i*8+j].permute(1, 2, 0).numpy())
    #         plt.axis("off")
    # plt.show()
    # normalize the images in [-1,1] from [0, 255]
    images = (images - 127.5) / 127.5
    outputs = wr(images.to(device))
    class_predictions = torch.argmax(outputs, dim=1)
    for p in class_predictions:
        hist[cifar10_classes[p]] += 1


# plot the histogram
import matplotlib.pyplot as plt

plt.bar(hist.keys(), hist.values())
plt.show()

# %%
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

# plot the histogram
import matplotlib.pyplot as plt

plt.bar(generated_img_histogram.keys(), generated_img_histogram.values())
plt.show()
# %%
# plot all the images in a grid 10 x 10
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(images_to_be_evaluated[i * 10 + j])
        axes[i, j].axis("off")
plt.show()
# %%
shape = (100,) + input_shape
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

fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(images_to_be_evaluated[i * 10 + j])
        axes[i, j].axis("off")
plt.show()
# %%
img = x.float().to(device)
print(img.shape)

batch = next(iter(test_dataloader))
print(batch.shape)

# plot the image of the batch in a grid 8 x 8
plt.figure(figsize=(20, 20))
for i in range(8):
    for j in range(8):
        plt.subplot(8, 8, i * 8 + j + 1)
        plt.imshow(batch[i * 8 + j].permute(1, 2, 0).numpy())
        plt.axis("off")
plt.show()

# %%
batch = next(iter(test_dataloader))

images = batch.to(device)
# plot the image of the batch in a grid 8 x 8
plt.figure(figsize=(20, 20))
for i in range(8):
    for j in range(8):
        plt.subplot(8, 8, i * 8 + j + 1)
        plt.imshow(images[i * 8 + j].permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
plt.show()

#  %%
images = diffusion.p_sample(
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

for i, im in enumerate(images):
    dists = []
    # plot the image
    plt.imshow(im.permute(1, 2, 0).cpu().numpy())

    emb_im = wr.penultimate_embedding(im.unsqueeze(0).to(device).float())

    for batch in test_dataloader:

        for image in batch:
            emb = wr.penultimate_embedding(image.to(device).float().unsqueeze(0))

            # evaluate the distance between the penultimate embedding of the generated image and the penultimate embedding of the dataset
            distance = torch.nn.functional.pairwise_distance(emb_im, emb)
            dists.append((distance, image.cpu()))
            #  sort the distances
            dists.sort(key=lambda x: x[0], reverse=False)
            # get the top 10 images
            dists = dists[:10]
    # get all the values of the distances
    ds = [d.item() for d, _ in dists]
    images_ds = [i for _, i in dists]
    print(ds)
    # plot all the images in a grid 1 x 11
    fig, axes = plt.subplots(1, 11, figsize=(20, 20))
    axes[0].imshow(images_to_be_evaluated[i])
    axes[0].axis("off")
    for i, img in enumerate(images_ds):

        axes[i + 1].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i + 1].axis("off")
    plt.show()


# %%
