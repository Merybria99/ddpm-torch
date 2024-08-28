# %%
import json
import os
import torch
from tqdm import tqdm
from ddim import DDIM, get_selection_schedule
from ddpm_torch import *
import argparse
import csv

model_instanciations = {
    "unet": UNet,
    "unet2h": UNet2H,
}


def eval_checkpoints(
    interval=50,
    model_arch="unet",
    dataset="cifar10",
    ckpt_dir="checkpoints",
    config_path="./configs/cifar10.json",
    config_dir="./configs",
    use_ddim=False,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset preprocessing
    if config_path is None:
        config_path = os.path.join(config_dir, dataset + ".json")
    with open(config_path, "r") as f:
        meta_config = json.load(f)

    dataset = meta_config.get("dataset", dataset)
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
    if use_ddim:
        diffusion_kwargs["model_var_type"] = "fixed-small"

        skip_schedule = kwargs.get("skip_schedule", 1)
        eta = kwargs.get("eta", 1e-4)
        subseq_size = kwargs.get("subseq_size", 1)

        skip_schedule = skip_schedule
        eta = eta
        subseq_size = subseq_size
        subsequence = get_selection_schedule(
            skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps
        )
        diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)
    else:
        diffusion = GaussianDiffusion(betas, **diffusion_kwargs)
    block_size = meta_config["model"].pop("block_size", 1)
    model = model_instanciations[model_arch](
        out_channels=in_channels, **meta_config["model"]
    )
    if block_size > 1:
        pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
        post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
        model = ModelWrapper(model, pre_transform, post_transform)
    model.to(device)
    use_ema = meta_config["train"].get("use_ema", False)

    # log the metrics a file
    log_dir = kwargs.get("log_dir", f"{ckpt_dir}/logs")
    os.makedirs(log_dir, exist_ok=True)

    # write the header
    with open(os.path.join(log_dir, f"metrics.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "FID", "IS"])

    for chkpt in os.listdir(ckpt_dir):
        if not chkpt.endswith(".pt"):
            continue
        chkpt_path = os.path.join(ckpt_dir, chkpt)
        epoch = int(chkpt.split("_")[-1].split(".")[0])
        if epoch % interval != 0:
            continue
        
        print(f"Processing checkpoint {chkpt} at epoch {epoch}")
        
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

        # load the dataset
        dataset_train = DATASET_DICT[dataset](
            "/home/maria.briglia/data/ddpm-train", "train"
        )
        torch.manual_seed(0)
        batch_size = 100
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )

        # instantiate the FID and IS
        from torchmetrics.image import FrechetInceptionDistance, InceptionScore

        fid = FrechetInceptionDistance()
        inception_score = InceptionScore()

        # iterate over the dataset for 10k samples and compute the FID and IS
        tot_samples = 10_000
        for i, x in enumerate(train_dataloader):
            if i * batch_size > tot_samples:
                break
            fid.update(x, real=True)

        # generate samples from the model
        pbar = None
        num_batches = tot_samples // batch_size
        pbar = tqdm(total=num_batches)

        for i in range(num_batches):
            shape = (batch_size, 3, image_res, image_res)
            x = diffusion.p_sample(
                model,
                shape=shape,
                device=device,
                noise=torch.randn(shape, device=device),
            ).cpu()
            print(x.shape)
            x = (
                (x * 127.5 + 127.5)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )

            pbar.update(1)
            fid.update(x, real=False)
            inception_score.update(x)
        # update the metrics csv file
        with open(os.path.join(log_dir, f"metrics.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, fid.compute().item(), inception_score.compute().item()])
        pbar.close()


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval", type=int, default=50, help="interval between checkpoints"
    )
    parser.add_argument(
        "--model-arch", type=str, default="unet", help="model architecture"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset to use")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="/home/maria.briglia/data/ddpm-train/chkpts/cifar10",
        help="directory of checkpoints",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="/home/maria.briglia/AdvTrain/ddpm-torch/configs/cifar10.json",
        help="path to the configuration file",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="/home/maria.briglia/AdvTrain/ddpm-torch/configs",
        help="directory of configuration files",
    )
    parser.add_argument("--use-ddim", action="store_true", help="use DDIM for sampling")
    parser.add_argument(
        "--skip-schedule", type=str, default="linear", help="skip schedule"
    )
    parser.add_argument("--eta", type=float, default=1e-4, help="eta")
    parser.add_argument(
        "--subseq-size", type=int, default=1, help="subsequence size for DDIM"
    )
    parser.add_argument("--log-dir", type=str, default="./logs", help="log directory")

    args = parser.parse_known_args()[0]
    # eval_checkpoints(
    #     interval=args.interval,
    #     model_arch=args.model_arch,
    #     dataset=args.dataset,
    #     ckpt_dir=args.ckpt_dir,
    #     config_path=args.config_path,
    #     config_dir=args.config_dir,
    #     use_ddim=args.use_ddim,
    #     skip_schedule=args.skip_schedule,
    #     eta=args.eta,
    #     subseq_size=args.subseq_size,
    #     # log_dir=args.log_dir,
    # )

# %%
