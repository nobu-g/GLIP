import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image, ImageFile
from rhoknp import KNP, Document
from transformers import BatchEncoding
from yacs.config import CfgNode

from tools.chunk_aligner import align_chunks
from tools.util import CamelCaseDataClassJsonMixin, Rectangle, get_core_expression

torch.set_grad_enabled(False)


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    image_id: str
    rect: Rectangle
    confidence: float


@dataclass(frozen=True)
class PhrasePrediction(CamelCaseDataClassJsonMixin):
    index: int
    text: str
    bounding_boxes: List[BoundingBox]


@dataclass(frozen=True)
class GLIPPrediction(CamelCaseDataClassJsonMixin):
    doc_id: str
    image_id: str
    phrases: List[PhrasePrediction]


# for output bounding box post-processing
def box_cxcywh_to_xyxy(
    x: torch.Tensor,  # (N, 4)
) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(
    out_bbox: torch.Tensor,  # (N, 4)
    size: Tuple[int, int],
) -> torch.Tensor:  # (N, 4)
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image


def plot_results(
    image: ImageFile, prediction: GLIPPrediction, output_file: Path, confidence_threshold: float = 0.8
) -> None:
    plt.figure(figsize=(16, 10))
    np_image = np.array(image)
    ax = plt.gca()
    colors = COLORS * 100

    for phrase_prediction in prediction.phrases:
        for bounding_box in phrase_prediction.bounding_boxes:
            rect = bounding_box.rect
            score = bounding_box.confidence
            if score < confidence_threshold:
                continue
            label = phrase_prediction.text
            color = colors.pop()
            ax.add_patch(plt.Rectangle((rect.x1, rect.y1), rect.w, rect.h, fill=False, color=color, linewidth=3))
            ax.text(
                rect.x1,
                rect.y1,
                f"{label}: {score:0.2f}",
                fontsize=15,
                bbox=dict(facecolor=color, alpha=0.8),
                # fontname="Hiragino Maru Gothic Pro",
                fontname="IPAexGothic",
            )

    plt.imshow(np_image)
    plt.axis("off")
    plt.savefig(output_file)
    # plt.show()


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


def predict_glip(cfg: CfgNode, images: list, image_ids: list[str], caption: Document) -> List[GLIPPrediction]:
    if len(images) == 0:
        return []

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.0,
        show_mask_heatmaps=False,
    )
    encoded: BatchEncoding = glip_demo.tokenizer(caption.text).encodings[0]

    assert caption.is_jumanpp_required() is False

    custom_entity: List[List[List]] = []
    char_index = 0
    for base_phrase in caption.base_phrases:
        if base_phrase.features.get("体言") is True:
            before, core, after = get_core_expression(base_phrase)
            core_start = char_index + len(before)
            core_end = core_start + len(core)
            custom_entity.append([[core_start, core_end]])
        char_index += len(base_phrase.text)

    chunks_a = []
    for token_index in range(len(encoded.tokens)):
        char_span = encoded.token_to_chars(token_index)
        if char_span is None or encoded.tokens[token_index] == "▁":
            chunks_a.append(0)
            continue
        char_start_index, char_end_index = char_span
        chunks_a.append(char_end_index - char_start_index)
    chunks_b = []
    for base_phrase in caption.base_phrases:
        chunks_b.append(len("".join(base_phrase.text.split("\u3000"))))
    token_base_phrase_alignments = align_chunks(chunks_a, chunks_b)

    if cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
        plus = 1
    else:
        plus = 0

    predictions: List[GLIPPrediction] = []
    assert len(images) == len(image_ids)
    for image, image_id in zip(images, image_ids):
        # convert to BGR format
        numpy_image = np.array(image)[:, :, [2, 1, 0]]
        output: BoxList = glip_demo.inference(numpy_image, caption.text, custom_entity=custom_entity)
        positive_map_label_to_token: Dict[int, List[int]] = glip_demo.positive_map_label_to_token
        boxes_list, scores_list = flickr_post_process(output, positive_map_label_to_token, plus)

        phrase_index_to_bounding_boxes: Dict[int, List[BoundingBox]] = defaultdict(list)
        assert (
            len(boxes_list) == len(scores_list) == len(positive_map_label_to_token)
        ), f"{len(boxes_list)}, {len(scores_list)}, {len(positive_map_label_to_token)}"
        for boxes, scores, (_, token_indices) in zip(boxes_list, scores_list, positive_map_label_to_token.items()):
            bounding_boxes: List[BoundingBox] = [
                BoundingBox(
                    image_id=image_id,
                    rect=Rectangle(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    confidence=score,
                )
                for box, score in zip(boxes, scores)
            ]

            # ensure token_indeices are consecutive
            assert token_indices == list(range(token_indices[0], token_indices[-1] + 1))
            for token_index in token_indices:
                for base_phrase_index in token_base_phrase_alignments[token_index]:
                    phrase_index_to_bounding_boxes[base_phrase_index] += bounding_boxes

        phrase_predictions: List[PhrasePrediction] = [
            PhrasePrediction(
                index=base_phrase.global_index,
                text=base_phrase.text,
                bounding_boxes=phrase_index_to_bounding_boxes.get(base_phrase.global_index, []),
            )
            for base_phrase in caption.base_phrases
        ]

        predictions.append(
            GLIPPrediction(
                doc_id=caption.doc_id,
                image_id=image_id,
                phrases=phrase_predictions,
            )
        )
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/grounding/e2e_dyhead_SwinT_S_FPN_1x_od_grounding_eval.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--model", "-m", type=str, help="Path to trained model.")
    parser.add_argument("--image-files", "--img", type=str, nargs="*", help="Path to images files.")
    parser.add_argument(
        "--text", type=str, default="5 people each holding an umbrella", help="split text to perform grounding."
    )
    parser.add_argument("--caption-file", type=str, help="Path to Juman++ file for caption.")
    parser.add_argument("--text-encoder", type=str, default=None, help="text encoder name")
    # parser.add_argument("--batch-size", "--bs", type=int, default=32, help="Batch size.")
    parser.add_argument("--export-dir", type=str, help="Path to directory to export results.")
    parser.add_argument("--plot", action="store_true", help="Plot results.")
    args = parser.parse_args()

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

    export_dir = Path(args.export_dir)
    export_dir.mkdir(exist_ok=True)

    images: list = [Image.open(image_file).convert("RGB") for image_file in args.image_files]
    image_ids: list[str] = [Path(image_file).stem for image_file in args.image_files]
    if args.caption_file is not None:
        caption = Document.from_knp(Path(args.caption_file).read_text())
    else:
        knp = KNP(options=["-dpnd-fast", "-tab"])
        caption = knp.apply_to_document(args.text)

    predictions = predict_glip(cfg, images, image_ids, caption)
    assert len(predictions) == len(images)
    if args.plot:
        for prediction, image, image_id in zip(predictions, images, image_ids):
            plot_results(image, prediction, export_dir / f"{image_id}.png", confidence_threshold=0.3)

    for image_id, prediction in zip(image_ids, predictions):
        export_dir.joinpath(f"{image_id}.json").write_text(prediction.to_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
