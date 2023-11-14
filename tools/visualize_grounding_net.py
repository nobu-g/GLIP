# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import argparse
import datetime
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.distributed as dist
from dataclasses_json import DataClassJsonMixin, LetterCase, config
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from PIL import Image
from tools.util import CamelCaseDataClassJsonMixin, Rectangle
from transformers import BatchEncoding

# from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
# from maskrcnn_benchmark.utils.comm import get_rank, synchronize

pylab.rcParams["figure.figsize"] = 20, 12


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    rect: Rectangle
    class_name: str
    confidence: float


@dataclass(frozen=True)
class PhrasePrediction(CamelCaseDataClassJsonMixin):
    phrase_index: int
    phrase: str
    bounding_boxes: List[BoundingBox]


@dataclass(frozen=True)
class GLIPPrediction(CamelCaseDataClassJsonMixin):
    image_id: int
    sentence_id: int
    phrase_predictions: List[BoundingBox]
    phrases: List[str]


def load_image(url_or_file_name: str) -> np.ndarray:
    try:
        response = requests.get(url_or_file_name)
    except:
        response = None
    if response is None:
        pil_image = Image.open(url_or_file_name).convert("RGB")
    else:
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    # args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(0, 7200),
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def imshow(img: np.ndarray, file_name: Union[str, Path] = "tmp.jpg"):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, "test", wrap=True, horizontalalignment="center", fontsize=20)
    plt.savefig(str(file_name))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument(
        "--config-file",
        default="configs/grounding/e2e_dyhead_SwinT_S_FPN_1x_od_grounding_eval.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        default=None,
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--task_config", default=None)

    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--box_pixel", default=3, type=int)
    parser.add_argument("--text_size", default=1, type=float)
    parser.add_argument("--text_pixel", default=1, type=int)
    parser.add_argument("--image_index", default=0, type=int)
    parser.add_argument("--threshold", default=0.6, type=float)
    parser.add_argument("--text_offset", default=10, type=int)
    parser.add_argument("--text_offset_original", default=4, type=int)
    parser.add_argument("--color", default=255, type=int)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # distributed = num_gpus > 1

    # cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    log_dir = cfg.OUTPUT_DIR
    if args.weight:
        log_dir = os.path.join(log_dir, "eval", os.path.splitext(os.path.basename(args.weight))[0])
    if log_dir:
        mkdir(log_dir)

    logger = setup_logger("maskrcnn_benchmark", log_dir, distributed_rank=0)
    logger.info(args)
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(cfg)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    if args.weight:
        _ = checkpointer.load(args.weight, force=True)
    else:
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

    all_task_configs = args.task_config.split(",")
    for task_config in all_task_configs:
        cfg_ = cfg.clone()
        cfg_.defrost()
        cfg_.merge_from_file(task_config)
        cfg_.merge_from_list(args.opts)
        assert cfg_.TEST.IMS_PER_BATCH == 1
        iou_types = ("bbox",)
        if cfg_.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg_.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        dataset_names = cfg_.DATASETS.TEST
        assert len(dataset_names) == 1
        dataset_name = dataset_names[0]
        output_folder = os.path.join(log_dir, "inference", dataset_name)

        data_loaders_val = make_data_loader(cfg_, is_train=False)
        assert len(data_loaders_val) == 1
        data_loader_val = data_loaders_val[0]

        # visualizer = GLIPDemo(
        #     cfg,
        #     min_image_size=800,
        #     confidence_threshold=0.7,
        #     show_mask_heatmaps=False,
        #     load_model=False,
        # )

        mdetr_style_output = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=(cfg_.MODEL.RPN_ONLY and (cfg_.MODEL.RPN_ARCHITECTURE == "RPN" or cfg_.DATASETS.CLASS_AGNOSTIC)),
            device=cfg_.MODEL.DEVICE,
            expected_results=cfg_.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg_.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            cfg=cfg_,
            # visualizer=visualizer,
        )
        if cfg_.TEST.SUBSET > 0:
            assert len(mdetr_style_output) == cfg_.TEST.SUBSET
        else:
            assert len(mdetr_style_output) == len(data_loader_val)
        for prediction in mdetr_style_output:
            image_id: int = prediction["image_id"]
            sentence_id: int = prediction["sentence_id"]
            boxes_list: List[List[List[float]]] = prediction["boxes"]
            scores_list: List[List[float]] = prediction["scores"]
            caption: str = prediction["caption"]
            encoded: BatchEncoding = model.tokenizer(caption).encodings[0]
            positive_map: Dict[int, List[int]] = prediction["positive_map"]
            assert (
                len(boxes_list) == len(scores_list) == len(positive_map)
            ), f"{len(boxes_list)}, {len(scores_list)}, {len(positive_map)}"
            # import ipdb; ipdb.set_trace()
            current_token_index = 1  # skip the [CLS] token
            current_phrase_index = 0
            phrase_predictions = []
            for boxes, scores, (_, token_indices) in zip(boxes_list, scores_list, positive_map.items()):
                # ensure token_indeices are consecutive
                assert token_indices == list(range(token_indices[0], token_indices[-1] + 1))
                if token_indices[0] > current_token_index:
                    char_start_index = encoded.token_to_chars(current_token_index)[0]
                    char_end_index = encoded.token_to_chars(token_indices[0])[0]
                    # token_ids = encoded.ids[current_token_index:token_indices[0]]
                    phrase_predictions.append(
                        PhrasePrediction(
                            phrase_index=current_phrase_index,
                            phrase=caption[char_start_index:char_end_index],
                            bounding_boxes=[],
                        )
                    )
                    current_phrase_index += 1
                    current_token_index = token_indices[0]
                else:
                    assert token_indices[0] == current_token_index

                bounding_boxes: List[BoundingBox] = []
                for box, score in zip(boxes, scores):
                    bounding_boxes.append(
                        BoundingBox(
                            rect=Rectangle(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                            class_name="",
                            confidence=score,
                        )
                    )

                char_start_index = encoded.token_to_chars(token_indices[0])[0]
                char_end_index = encoded.token_to_chars(token_indices[-1])[1]
                phrase_predictions.append(
                    PhrasePrediction(
                        phrase_index=current_phrase_index,
                        phrase=caption[char_start_index:char_end_index],
                        bounding_boxes=bounding_boxes,
                    )
                )
                current_phrase_index += 1
                current_token_index = token_indices[-1] + 1

            glip_prediction = GLIPPrediction(
                image_id=image_id,
                sentence_id=sentence_id,
                phrase_predictions=phrase_predictions,
                phrases=[pp.phrase for pp in phrase_predictions],
            )
            import ipdb

            ipdb.set_trace()
            print(glip_prediction)

        # dataset = data_loader_val.dataset
        # image_id = dataset.ids[image_index]
        # image_path = os.path.join(dataset.root, dataset.coco.loadImgs(image_id)[0]["file_name"])
        # categories = dataset.coco.dataset["categories"]

        # image = load_image(image_path)
        # no_background = True
        # label_list = []
        # for index, cat in enumerate(categories):
        #     # assert(index + 1 == cat["id"])
        #     if not no_background or (cat["name"] != "__background__" and cat["id"] != 0):
        #         label_list.append(cat["name"])
        # visualizer.entities = label_list

        # result, _ = visualizer.visualize_with_predictions(
        #     image,
        #     predictions,
        #     thresh=args.threshold,
        #     alpha=args.alpha,
        #     box_pixel=args.box_pixel,
        #     text_size=args.text_size,
        #     text_pixel=args.text_pixel,
        #     text_offset=args.text_offset,
        #     text_offset_original=args.text_offset_original,
        #     color=args.color,
        # )
        # visualize_dir = Path("./visualize")
        # visualize_dir.mkdir(exist_ok=True)
        # imshow(result, visualize_dir / f"tmp{image_index}.jpg")
        # image_index += 1


if __name__ == "__main__":
    main()
