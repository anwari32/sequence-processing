import json
from models.seq2seq import DNABERTSeq2Seq

def init_seq2seq_model(pretrained_path, config: json):
    model = DNABERTSeq2Seq(
        pretrained_path,
        config["num_blocks"],
        config["num_labels"],
        config["hidden_dim"],
        not (config["norm_layer"] == 0),
        config["dropout_prob"]
    )
    return model

