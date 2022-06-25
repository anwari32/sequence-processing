# sequence-processing
Experiment on biological sequence with machine learning or any other methods. If you are using this repository as a reference, please cite properly.

# sequential labelling training command
## sequential labelling with small dataset, batch size = 64, num epochs = 50, base model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-base-b64-small-e50 --num-epochs=50

## sequential labelling with small dataset, batch size = 64, num epochs = 50, base with dropout model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/base.drop.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-basedrop-b64-small-e50 --num-epochs=50


# gene sequential labelling training command
## gene sequential labelling with small dataset, num epochs = 50, base model
## gene sequential labelling with small dataset, num epochs = 50, base with dropout model