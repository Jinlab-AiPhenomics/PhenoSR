import argparse
import cv2
import glob
import os
from tqdm import tqdm


def main(args):
    txt_file = open(os.path.join(args.save_path, "meta_info.txt"), "w")

    parent_path = [os.path.dirname(args.input)]
    hr_path = [args.input]

    for folder, root in zip(hr_path, parent_path):
        img_paths = sorted(glob.glob(os.path.join(folder, "*")))
        for img_path in tqdm(img_paths):
            status = True
            if status:
                img_name = os.path.relpath(img_path, root)
                txt_file.write(f"{img_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Input folder",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="txt path to save meta info file",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    main(args)
