# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch
from maskrcnn_benchmark.utils.big_model_loading import load_big_format
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.model_zoo import cache_url
from maskrcnn_benchmark.utils.pretrain_model_loading import load_pretrain_format


class Checkpointer:
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            if isinstance(self.scheduler, list):
                data["scheduler"] = [scheduler.state_dict() for scheduler in self.scheduler]
            else:
                data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, f"{name}.pth")
        self.logger.info(f"Saving checkpoint to {save_file}")
        torch.save(data, save_file)
        # self.tag_last_checkpoint(save_file)
        # use relative path name to save the checkpoint
        self.tag_last_checkpoint(f"{name}.pth")

    def load(self, f=None, force=False, keyword="model", skip_optimizer=False):
        resume = False
        if self.has_checkpoint() and not force:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
            # get the absolute path
            f = os.path.join(self.save_dir, f)
            resume = True
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info(f"Loading checkpoint from {f}")
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, keyword=keyword)
        # if resume training, load optimizer and scheduler,
        # otherwise use the specified LR in config yaml for fine-tuning
        if resume and not skip_optimizer:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info(f"Loading optimizer from {f}")
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info(f"Loading scheduler from {f}")
                if isinstance(self.scheduler, list):
                    for scheduler, state_dict in zip(self.scheduler, checkpoint.pop("scheduler")):
                        scheduler.load_state_dict(state_dict)
                else:
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

            # return any further checkpoint data
            return checkpoint
        else:
            return {}

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file) as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except OSError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, keyword="model"):
        load_state_dict(self.model, checkpoint.pop(keyword))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super().__init__(model, optimizer, scheduler, save_dir, save_to_disk, logger)
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True)
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info(f"{f} points to {catalog_f}")
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info(f"url {f} cached in {cached_f}")
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        if f.endswith(".big"):
            return load_big_format(self.cfg, f)
        if f.endswith(".pretrain"):
            return load_pretrain_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super()._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
