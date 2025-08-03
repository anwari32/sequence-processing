# To compensate my own doing in creating model from scratch while there is easier way. 
# I hate this.

from transformers import BertPreTrainedModel, BertModel
from ...__models.seqlab import Head
from torch.nn import Softmax

class DNABERT_SL(BertPreTrainedModel):
    def __init__(self, config, additional_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        self.head = Head(additional_config)
        self.activation = Softmax(dim=2)

        self.post_init()
    
    def forward(self, input_ids, attention_masks):
        # output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks)            
        bert_output = output[0]
        output = output[0]
        output = self.head(output)
        head_output = output
        output = self.activation(output)
        return output, bert_output, head_output