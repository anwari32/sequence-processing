# ensemble implementation

import numpy as np

class Ensemble():
    def __init__(self, pred_bilstm=None, pred_bigru=None, pred_dnabert=None, num_classes=8):
        self.bilstm_prediction = pred_bilstm
        self.bigru_prediction = pred_bigru
        self.dnabert_prediction = pred_dnabert
        self.num_classes = num_classes
        self.final_prediction = []

    def consolidate(self):
        for i in range(len(self.bilstm_prediction)):
            label_vote = []
            for k in range(self.num_classes):
                label_vote.append(0)
            label_vote[self.bilstm_prediction[i]] += 1
            label_vote[self.bigru_prediction[i]] += 1
            label_vote[self.dnabert_prediction[i]] += 1
            max = np.argmax(label_vote)
            self.final_prediction.append(max)

        return np.array(self.final_prediction)

if __name__ == "__main__":
    num_classes = 8
    s1 = np.array([1,2,3,7,7,7,7,4,1,1,3])
    s2 = np.array([1,3,5,4,5,4,3,4,5,4,3])
    s3 = np.array([2,2,3,4,5,4,3,4,5,4,3])

    ensemble = Ensemble(s1, s2, s3, num_classes=num_classes)
    f = ensemble.consolidate()
    print(s1)
    print(s2)
    print(s3)
    print(f)


    

