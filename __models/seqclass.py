# implements DNABERT for sequence labelling.
num_classes = 3
id2label = {
    0: "negative-exon",
    1: "negative-intron",
    2: "positive"
}

label2id = {
    "negative-exon": 0,
    "negative-intron": 1,
    "positive": 2
}

from torch import nn
from transformers import BertForSequenceClassification, BertModel
import os
import json

class DNABertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids, attention_mask):
        output = super().forward(input_ids, attention_mask)
        return output

if __name__ == "__main__":
    default_config = json.load(
        open(
            os.path.join("pretrained", "3-new-12w-0", "config.json"),
            "r"
        )
    )
    model = DNABertForSequenceClassification(default_config)
    print(model)