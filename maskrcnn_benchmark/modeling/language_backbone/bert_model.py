from copy import deepcopy

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_type = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        print(
            "LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ",
            self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT,
        )

        config = AutoConfig.from_pretrained(self.model_type)
        config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
        if "deberta" in self.model_type:
            self.model = AutoModel.from_pretrained(self.model_type, config=config)
        else:
            self.model = AutoModel.from_pretrained(self.model_type, add_pooling_layer=False, config=config)
        self.num_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # with padding, always 256
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]
            features = torch.stack(encoded_layers[-self.num_layers :], 1).mean(1)

            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers

            embedded = features * attention_mask.unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (attention_mask.sum(-1).unsqueeze(-1).float())

        else:
            # without padding, only consider positive_tokens
            max_len = (input_ids != 0).sum(1).max().item()
            outputs = self.model(
                input_ids=input_ids[:, :max_len],
                attention_mask=attention_mask[:, :max_len],
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]

            features = torch.stack(encoded_layers[-self.num_layers :], 1).mean(1)
            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers

            embedded = features * attention_mask[:, :max_len].unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (attention_mask.sum(-1).unsqueeze(-1).float())

        ret = {
            "aggregate": aggregate,
            "embedded": embedded,
            "masks": attention_mask,
            "hidden": encoded_layers[-1],
        }
        return ret
