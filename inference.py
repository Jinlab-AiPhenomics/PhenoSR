import cv2
import glob
import os
import torch
import argparse
from basicsr.archs.phenosr_arch import PhenoSR
from utils import Generator
import shutil
from tqdm import tqdm
from pyiqa import create_metric
from colorama import Back, Style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.1, help="se_score threshold"
    )
    parser.add_argument(
        "-c", "--calculate", type=bool, default=True, help="calculate metrics"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="./weights/phenosr_20240831.pth",
        help="model path",
    )
    parser.add_argument(
        "-i", "--input", type=str, default="./input", help="input folder"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="./output",
        help="output folder",
    )
    args = parser.parse_args()

    if args.calculate:
        total_niqe_sr = 0
        total_hyperiqa = 0
        niqe_iqa_model = create_metric("niqe")
        hyperiqa_iqa_model = create_metric("hyperiqa")

    model = PhenoSR(
        img_size=64,
        upscale=4,
        in_chans=3,
        window_size=8,
        img_range=1.0,
        seg_dim=32,
        coarse_depths=(6, 6),
        coarse_num_heads=(6, 6),
        refine_depths=(6, 6, 6, 6),
        refine_num_heads=(6, 6, 6, 6),
        embed_dim=180,
        num_class=2,
        mlp_ratio=2,
        threshold=args.threshold,
    ).cuda()

    output_path = f"./{args.out}/sr"
    seg_output_path = f"./{args.out}/seg"

    upsampler = Generator(scale=4, model_path=args.model_path, model=model)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if os.path.exists(seg_output_path):
        shutil.rmtree(seg_output_path)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(seg_output_path, exist_ok=True)

    paths = (
        [args.input]
        if os.path.isfile(args.input)
        else sorted(glob.glob(os.path.join(args.input, "*")))
    )
    save_results = {}

    for path in tqdm(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_mode = "RGBA" if len(img.shape) == 3 and img.shape[2] == 4 else None

        try:
            sr_out, seg_out = upsampler.enhance(img, outscale=4, imgname=imgname)
            seg_map = torch.argmax(seg_out, dim=1, keepdim=True)
        except RuntimeError as error:
            print("Error", error)
        else:
            extension = "png" if img_mode == "RGBA" else extension[1:]
            save_path = os.path.join(output_path, f"{imgname}_out.{extension}")

            cv2.imwrite(save_path, sr_out)
            cv2.imwrite(
                os.path.join(seg_output_path, f"{imgname}_seg.png"),
                seg_map[0].cpu().numpy().transpose(1, 2, 0) * 255,
            )
            if args.calculate:
                current_niqe_sr = niqe_iqa_model(save_path, None).cpu().item()
                current_hyperiqa = hyperiqa_iqa_model(save_path, None).cpu().item()

                total_niqe_sr += current_niqe_sr
                total_hyperiqa += current_hyperiqa

                save_results[imgname] = {
                    "niqe": current_niqe_sr,
                    "hyperiqa": current_hyperiqa,
                }

    if args.calculate:
        print(Back.RED)
        print("Threshold:", args.threshold)
        print(f"Average NIQE: {total_niqe_sr / len(paths):.4f}")
        print(f"Average HYPERIQA: {total_hyperiqa / len(paths):.4f}")
        print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
