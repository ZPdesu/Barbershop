import os
import dlib
import wandb
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torchvision

from models.Embedding import Embedding
from models.Alignment import Alignment
from models.Blending import Blending

from utils.drive import open_url
from utils.shape_predictor import align_face


def apply_align_faces(configs, identity_image):
    cache_dir = Path(configs.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(configs.align_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Shape Predictor")
    f = open_url(
        "https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx",
        cache_dir=cache_dir,
        return_path=True,
    )
    predictor = dlib.shape_predictor(f)

    identity_image = Path(identity_image)

    faces = align_face(str(identity_image), predictor)

    for i, face in enumerate(faces):
        if configs.output_size:
            factor = 1024 // configs.output_size
            assert configs.output_size * factor == 1024
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
            face = torchvision.transforms.ToPILImage()(face_tensor_lr)
            if factor != 1:
                face = face.resize(
                    (configs.output_size, configs.output_size), Image.LANCZOS
                )
        if len(faces) > 1:
            face.save(Path(configs.align_output_dir) / (identity_image.stem + f"_{i}.png"))
        else:
            face.save(Path(configs.align_output_dir) / (identity_image.stem + f".png"))


def main(args):
    wandb.login()
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="test",
        config=vars(args),
    ):

        images_artifact = wandb.use_artifact(args.images_artifact, type="dataset")
        images_artifact_dir = images_artifact.download()

        ffhq_model_artifact = wandb.use_artifact(
            args.ffhq_models_artifact, type="model"
        )
        ffhq_model_artifact_dir = ffhq_model_artifact.download()
        ffhq_model_file = os.path.join(ffhq_model_artifact_dir, "ffhq.pt")

        segmentation_model_artifact = wandb.use_artifact(
            args.segmentation_models_artifact, type="model"
        )
        segmentation_model_artifact_dir = segmentation_model_artifact.download()
        segmentation_model_file = os.path.join(
            segmentation_model_artifact_dir, "seg.pth"
        )

        identity_image = os.path.join(images_artifact_dir, args.identity_image)
        structure_image = os.path.join(images_artifact_dir, args.structure_image)
        appearance_image = os.path.join(images_artifact_dir, args.appearance_image)

        apply_align_faces(args, identity_image=identity_image)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        ii2s = Embedding(args, checkpoint_file=ffhq_model_file).to(device=device)

        im_set = {identity_image, structure_image, appearance_image}
        ii2s.invert_images_in_W([*im_set])
        ii2s.invert_images_in_FS([*im_set])

        align = Alignment(
            args,
            ffhq_checkpoint_file=ffhq_model_file,
            segmentation_checkpoint_file=segmentation_model_file,
        ).to(device=device)
        aligned_image = align.align_images(
            identity_image,
            structure_image,
            sign=args.sign,
            align_more_region=False,
            smooth=args.smooth,
        )
        if structure_image != appearance_image:
            aligned_image = align.align_images(
                identity_image,
                appearance_image,
                sign=args.sign,
                align_more_region=False,
                smooth=args.smooth,
                save_intermediate=False,
            )

        blend = Blending(
            args,
            ffhq_checkpoint_file=ffhq_model_file,
            segmentation_checkpoint_file=segmentation_model_file,
        ).to(device=device)
        blended_image = blend.blend_images(
            identity_image, structure_image, appearance_image, sign=args.sign
        )

        table_data = [[
            args.sign,
            wandb.Image(np.array(Image.open(identity_image))),
            wandb.Image(np.array(Image.open(structure_image))),
            wandb.Image(np.array(Image.open(appearance_image))),
            wandb.Image(np.array(aligned_image)),
            wandb.Image(np.array(blended_image)),
        ]]

        table = wandb.Table(
            data=table_data,
            columns=[
                "Realistic/Fidelity",
                "Identity-Image",
                "Structure-Image",
                "Appearance-Image",
                "Aligned-Image",
                "Blended-Image",
            ],
        )
        wandb.log({"Predictions": table})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Barbershop")

    # I/O arguments
    parser.add_argument(
        "--wandb_project", type=str, default="barbershop", help="WandB Project Name"
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB Entity")
    parser.add_argument(
        "--images_artifact",
        type=str,
        default="geekyrakshit/barbershop/II2S-Images:v0",
        help="WandB Artifact address for II2S Images",
    )
    parser.add_argument(
        "--ffhq_models_artifact",
        type=str,
        default="geekyrakshit/barbershop/ffhq:v0",
        help="WandB Artifact address for ffhq model",
    )
    parser.add_argument(
        "--segmentation_models_artifact",
        type=str,
        default="geekyrakshit/barbershop/segmentation:v0",
        help="WandB Artifact address for segmentation model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The directory to save the latent codes and inversion images",
    )
    parser.add_argument(
        "--identity_image", type=str, default="16.png", help="Identity image"
    )
    parser.add_argument(
        "--structure_image", type=str, default="15.png", help="Structure image"
    )
    parser.add_argument(
        "--appearance_image", type=str, default="117.png", help="Appearance image"
    )
    parser.add_argument(
        "--sign", type=str, default="realistic", help="realistic or fidelity results"
    )
    parser.add_argument(
        "--smooth", type=int, default=5, help="dilation and erosion parameter"
    )

    # Align Face Setting
    parser.add_argument(
        "--unprocessed_dir",
        type=str,
        default="unprocessed",
        help="directory with unprocessed images",
    )
    parser.add_argument(
        "--align_output_dir", type=str, default="input/face", help="output directory"
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=1024,
        help="size to downscale the input images to, must be power of 2",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="cache", help="cache directory for model weights"
    )

    # StyleGAN2 setting
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)

    # Arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--tile_latent",
        action="store_true",
        help="Whether to forcibly tile the same latent N times",
    )
    parser.add_argument(
        "--opt_name",
        type=str,
        default="adam",
        help="Optimizer to use in projected gradient descent",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate to use during optimization",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="fixed",
        help="fixed, linear1cycledrop, linear1cycle",
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Whether to store and save intermediate HR and LR images during optimization",
    )
    parser.add_argument(
        "--save_interval", type=int, default=300, help="Latent checkpoint interval"
    )
    parser.add_argument("--verbose", action="store_true", help="Print loss information")
    # parser.add_argument('--seg_ckpt', type=str, default='pretrained_models/seg.pth')

    # Embedding loss options
    parser.add_argument(
        "--percept_lambda",
        type=float,
        default=1.0,
        help="Perceptual loss multiplier factor",
    )
    parser.add_argument(
        "--l2_lambda", type=float, default=1.0, help="L2 loss multiplier factor"
    )
    parser.add_argument(
        "--p_norm_lambda",
        type=float,
        default=0.001,
        help="P-norm Regularizer multiplier factor",
    )
    parser.add_argument(
        "--l_F_lambda", type=float, default=0.1, help="L_F loss multiplier factor"
    )
    parser.add_argument(
        "--W_steps", type=int, default=1100, help="Number of W space optimization steps"
    )
    parser.add_argument(
        "--FS_steps", type=int, default=250, help="Number of W space optimization steps"
    )

    # Alignment loss options
    parser.add_argument(
        "--ce_lambda",
        type=float,
        default=1.0,
        help="cross entropy loss multiplier factor",
    )
    parser.add_argument(
        "--style_lambda", type=str, default=4e4, help="style loss multiplier factor"
    )
    parser.add_argument("--align_steps1", type=int, default=140, help="")
    parser.add_argument("--align_steps2", type=int, default=100, help="")

    # Blend loss options
    parser.add_argument("--face_lambda", type=float, default=1.0, help="")
    parser.add_argument("--hair_lambda", type=str, default=1.0, help="")
    parser.add_argument("--blend_steps", type=int, default=400, help="")

    args = parser.parse_args()
    main(args)
