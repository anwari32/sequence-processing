from torch import nn
from transformers import BertModel

Labels = [
    '...',
    '..E',
    '.E.',
    'E..',
    '.EE',
    'EE.',
    'E.E',
    'EEE',
]

Label_Dict = {
    '...': 0,
    '..E': 1,
    '.E.': 2,
    'E..': 3,
    '.EE': 4,
    'EE.': 5,
    'E.E': 6,
    'EEE': 7,
}

class DNASeqLabelling(nn.Module):
    """
    Core architecture of sequential labelling.
    """
    def __init__(self, bert_pretrained_path, device='cpu'):
        """
        Create instance of DNASeqLabelling. This model uses BERT as its feature extraction layer.
        This BERT layer is initiated from pretrained model which is located at `bert_pretrained_path`.
        @param  bert_pretrained_path (string):
        @param  device (string): default is 'cpu' but you can put 'cuda' if your machine supports cuda.
        @return (DNASeqLabelling): object of this class.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained_path)
        self.stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, input_ids, attention_masks):
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        output = output[0][:,0,:]
        output = self.stack(output)
        return output

    def train_and_validate(dataloader, optimizer=None, is_validate=True):
        return False
    
    def test(dataloader):
        return False

def train_and_validation(model, loss_function, optimizer, train_dataloader, validation_dataloader, epoch_size):
    raise NotImplementedError()

def test_sequential_labelling(model, test_dataloader):
    raise NotImplementedError()