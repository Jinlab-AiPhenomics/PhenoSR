import argparse
import glob
import os
from PIL import Image


def main(args):
    scale_list = [0.75, 0.5, 1 / 3]
    shortest_edge = 400

    path_list = sorted(glob.glob(os.path.join(args.input, "*")))
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        for idx, scale in enumerate(scale_list):
            print(f"\t{scale:.2f}")
            rlt = img.resize(
                (int(width * scale), int(height * scale)), resample=Image.LANCZOS
            )
            rlt.save(os.path.join(args.output, f"{basename}T{idx}.png"))

        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)
        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        rlt.save(os.path.join(args.output, f"{basename}T{idx+1}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input folder")
    parser.add_argument("-o", "--output", type=str, help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
