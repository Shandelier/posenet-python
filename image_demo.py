import tensorflow as tf
import cv2
import time
import argparse
import os
import numpy as np
import save_utils as su
from posenet.posenet_factory import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mobilenet")  # mobilenet resnet50
parser.add_argument(
    "--stride", type=int, default=16
)  # 8, 16, 32 (max 16 for mobilenet)
parser.add_argument("--quant_bytes", type=int, default=4)  # 4 = float
parser.add_argument("--multiplier", type=float, default=1.0)  # only for mobilenet
parser.add_argument("--notxt", action="store_true")
parser.add_argument("--image_dir", type=str, default="./images")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--pose", type=int, default=0)  # 0-straight, 1-slouche
args = parser.parse_args()


def main():
    print("Tensorflow version: %s" % tf.__version__)
    assert tf.__version__.startswith("2."), "Tensorflow version 2.x must be used!"

    if args.image_dir == args.output_dir:
        print(
            "[WARNING] input dir is the same as output dir -- the pictures will be overwritten"
        )
        print("Do you wish to continue?: y/n")
        if input() != "y":
            exit()

    # get input folder name to concatenate with out file name
    image_dir_name = os.path.basename(args.image_dir).split("./", 1)[0]
    # paths for posenet out pictures and .csv
    output_pic_dir_name = args.output_dir + "/" + image_dir_name
    output_csv_dir_name = args.output_dir + "/" + image_dir_name + ".csv"
    if not os.path.exists(output_pic_dir_name):
        os.makedirs(output_pic_dir_name)

    model = args.model  # mobilenet resnet50
    stride = args.stride  # 8, 16, 32 (max 16 for mobilenet, min 16 for resnet50)
    quant_bytes = args.quant_bytes  # float
    multiplier = args.multiplier  # only for mobilenet
    label = args.pose  # type of pose for pictures in folder

    posenet = load_model(model, stride, quant_bytes, multiplier)

    # file_count = su.count_files(args.image_dir)
    filenames = [
        f.path
        for f in os.scandir(args.image_dir)
        if f.is_file() and f.path.endswith((".png", ".jpg"))
    ]

    # prepare .csv file for points coords
    csv_file = open(output_csv_dir_name, "ab")
    np.savetxt(csv_file, su.csv_column_names, delimiter=",", fmt="%s")

    start = time.time()
    for f in filenames:
        img = cv2.imread(f)
        pose_scores, keypoint_scores, keypoint_coords = posenet.estimate_multiple_poses(
            img
        )
        img_poses = posenet.draw_poses(
            img, pose_scores, keypoint_scores, keypoint_coords
        )
        cv2.imwrite(
            os.path.join(output_pic_dir_name, os.path.relpath(f, args.image_dir)),
            img_poses,
        )

        one_row = np.concatenate(
            (
                [pose_scores[0]],  # certainity score for pose for pose 0 in 10 total
                keypoint_coords[0, :7, 0],  # X coords
                keypoint_coords[0, :7, 1],  # Y
                [label],  # pose type
                # TODO: picture number (first implement it in vid2pic)
            ),
            axis=0,
        ).reshape(su.csv_column_names.shape)
        np.savetxt(csv_file, one_row, delimiter=",")

    print("Average FPS:", len(filenames) / (time.time() - start))
    csv_file.close()


if __name__ == "__main__":
    main()
