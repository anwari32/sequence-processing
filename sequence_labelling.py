r"""
Sequence labelling class.
"""
class Sequence_Labelling():
    def __init__(self):
        self.model = None
        self.train_data = None
        self.train_dataloader = None
        self.validation_data = None
        self.validation_dataloader = None
        self.test_data = None
        self.test_dataloader = None
        self.learning_rate = None