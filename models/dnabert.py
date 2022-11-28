from transformers import BertForTokenClassification, BertForSequenceClassification, BertPreTrainedModel

num_classes = 8

class DNABertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output

class DNABertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output

class DNABertForTokenClassificationLong(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output
