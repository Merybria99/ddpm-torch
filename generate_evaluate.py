import json
import math
import os
import time
import torch
import torch.multiprocessing as mp
import uuid
from PIL import Image
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from ddim import DDIM, get_selection_schedule
from ddpm_torch import *
from multiprocessing.sharedctypes import Synchronized
from tqdm import tqdm
import numpy as np
import csv

# set the seed
torch.manual_seed(0)
np.random.seed(0)


def progress_monitor(total, counter):
    pbar = tqdm(total=total)
    while pbar.n < total:
        if pbar.n < counter.value:  # non-blocking intended
            pbar.update(counter.value - pbar.n)
        time.sleep(0.1)


def generate(rank, args, counter=0, **kwargs):
    assert isinstance(counter, (Synchronized, int))

    is_leader = rank == 0

    if args.config_path is None:
        args.config_path = os.path.join(args.config_dir, args.dataset + ".json")
    with open(args.config_path, "r") as f:
        meta_config = json.load(f)
    exp_name = kwargs.get("exp_name", os.path.basename(args.config_path)[:-5])

    dataset = meta_config.get("dataset", args.dataset)
    in_channels = DATASET_INFO[dataset]["channels"]
    image_res = DATASET_INFO[dataset]["resolution"][0]
    input_shape = (in_channels, image_res, image_res)

    diffusion_kwargs = meta_config["diffusion"]
    beta_schedule = diffusion_kwargs.pop("beta_schedule")
    beta_start = diffusion_kwargs.pop("beta_start")
    beta_end = diffusion_kwargs.pop("beta_end")
    num_diffusion_timesteps = diffusion_kwargs.pop("timesteps")
    betas = get_beta_schedule(
        beta_schedule, beta_start, beta_end, num_diffusion_timesteps
    )

    use_ddim = args.use_ddim
    print(f"Using DDIM: {use_ddim}")
    if use_ddim:
        diffusion_kwargs["model_var_type"] = "fixed-small"
        skip_schedule = args.skip_schedule
        eta = args.eta
        subseq_size = args.subseq_size
        subsequence = get_selection_schedule(
            skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps
        )
        diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)
    else:
        diffusion = GaussianDiffusion(betas, **diffusion_kwargs)

    device = torch.device(f"cuda:{rank}" if args.num_gpus > 1 else args.device)
    block_size = meta_config["model"].pop("block_size", 1)
    model = UNet(out_channels=in_channels, **meta_config["model"])

    if block_size > 1:
        pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
        post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
        model = ModelWrapper(model, pre_transform, post_transform)
    model.to(device)
    chkpt_dir = args.chkpt_dir
    chkpt_path = args.chkpt_path or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
    print(f"Loading checkpoint from {chkpt_path}...")

    folder_name = kwargs.get(
        "folder_name", os.path.basename(chkpt_path)[:-3]
    )  # truncated at file extension
    use_ema = meta_config["train"].get("use_ema", args.use_ema)

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

    folder_name = folder_name + args.suffix
    save_dir = kwargs.get(
        "save_dir", os.path.join(args.save_dir, "eval", exp_name, folder_name)
    )
    if is_leader and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    local_total_size = args.local_total_size
    batch_size = args.batch_size
    if args.world_size > 1:
        if rank < args.total_size % args.world_size:
            local_total_size += 1
    local_num_batches = math.ceil(local_total_size / batch_size)
    shape = (batch_size,) + input_shape

    def save_image(arr):
        with Image.fromarray(arr, mode="RGB") as im:
            im.save(f"{save_dir}/{uuid.uuid4()}.png")

    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa

    pbar = None
    if isinstance(counter, int):
        pbar = tqdm(total=local_num_batches)

    # check if the reverse chain is needed
    num_reverse = args.num_iterations if args.subsample_reverse_chain else None

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for i in range(local_num_batches):
            if i == local_num_batches - 1:
                shape = (local_total_size - i * batch_size, 3, image_res, image_res)
            x = diffusion.p_sample(
                model,
                shape=shape,
                device=device,
                noise=torch.randn(shape, device=device),
                performed=num_reverse,
            ).cpu()
            x = (
                (x * 127.5 + 127.5)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .numpy()
            )
            pool.map(save_image, list(x))
            if isinstance(counter, Synchronized):
                with counter.get_lock():
                    counter.value += 1
            else:
                pbar.update(1)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, help="path to the configuration file"
    )
    parser.add_argument("--dataset", choices=DATASET_DICT.keys(), default="cifar10")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--total-size", default=50000, type=int)
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-path", default="", type=str)
    parser.add_argument("--save-dir", default="./images", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--eta", default=0.0, type=float)
    parser.add_argument("--skip-schedule", default="linear", type=str)
    parser.add_argument("--subseq-size", default=50, type=int)
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--max-workers", default=8, type=int)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--interval", default=50, type=int)

    parser.add_argument(
        "--subsample-reverse-chain",
        action="store_true",
        help="subsample the reverse chain",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="number of iterations for the reverse chain",
    )

    args = parser.parse_args()

    world_size = args.world_size = args.num_gpus or 1
    local_total_size = args.local_total_size = args.total_size // world_size
    batch_size = args.batch_size
    remainder = args.total_size % world_size
    num_batches = math.ceil((local_total_size + 1) / batch_size) * remainder
    num_batches += math.ceil(local_total_size / batch_size) * (world_size - remainder)
    args.num_batches = num_batches
    exp_name = os.path.basename(args.config_path)[:-5]
    folder_name = os.path.basename(args.chkpt_path)[:-3]
    save_dir = os.path.join(args.save_dir, "eval", exp_name, folder_name)

    metrics_path = f"{args.chkpt_dir}/logs/metrics{'-DDIM' if args.use_ddim else '' }-{args.num_iterations}.csv"
    print(f"Metrics path: {metrics_path}")

    # check if the file exists
    epochs_computed = []
    if os.path.exists(metrics_path):
        # get the epochs already computed
        with open(metrics_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != "epoch":
                    epochs_computed.append(int(row[0]))
    print(f"Epochs already computed: {epochs_computed}")
    for file in os.listdir(args.chkpt_dir):
        if file.endswith(".pt"):
            epoch = int(file.split("_")[-1].split(".")[0])
            if (epoch % args.interval) != 0 or epoch in epochs_computed:
                continue
        else:
            continue
        if file.startswith(folder_name) and file.endswith(".pt"):
            args.chkpt_path = os.path.join(args.chkpt_dir, file)

        if not os.path.exists(save_dir):
            if world_size > 1:
                mp.set_start_method("spawn")
                counter = mp.Value("i", 0)
                mp.Process(
                    target=progress_monitor, args=(num_batches, counter), daemon=True
                ).start()
                mp.spawn(generate, args=(args, counter), nprocs=world_size)
            else:
                generate(
                    0,
                    args,
                    exp_name=exp_name,
                    folder_name=folder_name,
                    save_dir=save_dir,
                )

            print("Done for the generation!")

        # load the dataset
        dataset_train = DATASET_DICT[args.dataset](
            "/home/hl-fury/mariarosaria.briglia/ddpm-torch/data", "train"
        )
        torch.manual_seed(0)
        batch_size = 100
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )

        os.makedirs(f"{args.chkpt_dir}/logs", exist_ok=True)
        with open(
            metrics_path, "a"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "FID", "IS"])

        # instantiate the FID and IS
        from torchmetrics.image import FrechetInceptionDistance, InceptionScore

        print("Instanciating the FID and IS...")
        fid = FrechetInceptionDistance()
        inception_score = InceptionScore()

        # iterate over the dataset for 10k samples and compute the FID and IS
        tot_samples = args.total_size
        for i, x in enumerate(train_dataloader):
            if i * batch_size > tot_samples:
                break
            fid.update(x, real=True)

        # iterate over the generated samples and compute the FID and IS
        for image in os.listdir(save_dir):
            if not image.endswith(".png"):
                continue
            # load the image in a tensor
            x = Image.open(os.path.join(save_dir, image))
            x = torch.tensor(np.array(x)).permute(2, 0, 1).unsqueeze(0)
            fid.update(x, real=False)
            inception_score.update(x)

        with open(
            metrics_path, "a"
        ) as f:
            writer = csv.writer(f)
            fid = fid.compute()
            inception_score = inception_score.compute()[0]
            print(f"Epoch {epoch} - FID: {fid} - IS: {inception_score}")

            writer.writerow([epoch, fid, inception_score])
        # delete the generated samples
        for image in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, image))
        os.rmdir(save_dir)
        print("Done for the evaluation!")


if __name__ == "__main__":
    main()


"""
command lines for launching generation:
python generate_evaluate.py --config-path /home/hl-fury/mariarosaria.briglia/ddpm-torch/configs/cifar10.json --dataset cifar10 --total-size 10000  --use-ema --chkpt-dir /home/hl-fury/mariarosaria.briglia/ddpm-torch/models/adv-ddpm/L2/adv-post-2024-08-08T160539047657  --save-dir /home/hl-fury/mariarosaria.briglia/ddpm-torch/images/L2/images-evaluation --batch-size 512 
"""
