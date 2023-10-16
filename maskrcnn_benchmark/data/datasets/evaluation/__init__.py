from typing import List

from maskrcnn_benchmark.data import datasets
from maskrcnn_benchmark.structures.bounding_box import BoxList

from .box_aug import im_detect_bbox_aug
from .coco import coco_evaluation
from .od_to_grounding import od_to_grounding_evaluation
from .vg import vg_evaluation
from .voc import voc_evaluation


def evaluate(dataset, predictions: List[BoxList], output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs)
    if isinstance(dataset, datasets.COCODataset) or isinstance(dataset, datasets.TSVDataset):
        return coco_evaluation(**args)
    # elif isinstance(dataset, datasets.VGTSVDataset):
    #     return vg_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.CocoDetectionTSV):
        return od_to_grounding_evaluation(**args)
    elif isinstance(dataset, datasets.LvisDetection):
        pass
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError(f"Unsupported dataset type {dataset_name}.")


def evaluate_mdetr(dataset, predictions, output_folder, cfg):
    args = dict(dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs)
    if isinstance(dataset, datasets.COCODataset) or isinstance(dataset, datasets.TSVDataset):
        return coco_evaluation(**args)
    # elif isinstance(dataset, datasets.VGTSVDataset):
    #     return vg_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.CocoDetectionTSV):
        return od_to_grounding_evaluation(**args)
    elif isinstance(dataset, datasets.LvisDetection):
        pass
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError(f"Unsupported dataset type {dataset_name}.")
