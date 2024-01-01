import argparse
import os
import pickle
import tempfile
import warnings
from typing import Dict, List

import cv2
import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.structures.bounding_box import BoxList
from tqdm import tqdm

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/grounding/e2e_dyhead_SwinT_S_FPN_1x_od_grounding_eval.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--model", "-m", type=str, help="Path to trained model.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--text", type=str, default="もの", help="Text query for grounding.")
    parser.add_argument("--text-encoder", type=str, default=None, help="text encoder name")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom"],
        help="",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as temp_dir:
        filename = os.path.join(temp_dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def flickr_post_process(output: BoxList, positive_map_label_to_token: Dict[int, List[int]], plus: int) -> tuple:
    scores, indices = torch.topk(output.extra_fields["scores"], k=len(output.extra_fields["scores"]), sorted=True)
    boxes: List[List[float]] = output.bbox.tolist()
    boxes = [boxes[i] for i in indices]
    labels: List[int] = [output.extra_fields["labels"][i].item() for i in indices]
    output_boxes: List[List[List[float]]] = [[] for _ in range(len(positive_map_label_to_token))]
    output_scores: List[List[float]] = [[] for _ in range(len(positive_map_label_to_token))]
    for label, box, score in zip(labels, boxes, scores.tolist()):
        output_boxes[label - plus].append(box)
        output_scores[label - plus].append(score)

    return (
        output_boxes,  # (label, box, 4), label は対応する文に含まれるフレーズに振られたインデックス
        output_scores,  # (label, box)
    )


def main():
    args = parse_args()
    query_text = args.text
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(["MODEL.WEIGHT", args.model])
    if args.text_encoder is not None:
        cfg.merge_from_list(["MODEL.LANGUAGE_BACKBONE.MODEL_TYPE", args.text_encoder])
        cfg.merge_from_list(["MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE", args.text_encoder])
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.merge_from_list(["MODEL.DEVICE", device_str])
    cfg.freeze()

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.0,
        show_mask_heatmaps=False,
    )

    video = cv2.VideoCapture(args.video_input)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # basename = os.path.basename(args.video_input)
    codec, file_ext = ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    if codec == ".mp4v":
        warnings.warn("x264 codec not available, switching to mp4v", stacklevel=2)

    if cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
        plus = 1
    else:
        plus = 0

    export_obj = []
    frame: np.ndarray  # (h, w, 3), BGR
    for frame in tqdm(_frame_from_video(video), total=num_frames):
        output: BoxList = glip_demo.inference(frame, query_text, custom_entity=[[[0, len(query_text)]]])
        positive_map_label_to_token: Dict[int, List[int]] = glip_demo.positive_map_label_to_token
        boxes_list, scores_list = flickr_post_process(output, positive_map_label_to_token, plus)

        fields_list = []
        assert (
            len(boxes_list) == len(scores_list) == len(positive_map_label_to_token)
        ), f"{len(boxes_list)}, {len(scores_list)}, {len(positive_map_label_to_token)}"
        for box, score in zip(boxes_list[0], scores_list[0]):
            fields = (box[0], box[1], box[2], box[3], score, 0)  # x1, y1, x2, y2, score, class
            fields_list.append(fields)
        export_obj.append(np.array(fields_list))

    with open(args.output, mode="wb") as f:
        pickle.dump(export_obj, f)


if __name__ == "__main__":
    main()
