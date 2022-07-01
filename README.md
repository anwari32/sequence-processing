# sequence-processing
Experiment on biological sequence with machine learning or any other methods. If you are using this repository as a reference, please cite properly.

# sequential labelling training command

python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.tiny.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-tiny --num-epochs=50 --model-config-dir=models/config/seqlab --model-config-names=base,dropout,lin4,norm

## [BASE] tiny dataset, batch size = 64, num epochs = 50, base model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.tiny.json -m models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-base-b64-tiny-e50 --num-epochs=50

## [BASE] small dataset, batch size = 64, num epochs = 50, base model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-base-b64-small-e50 --num-epochs=50

## [DROPOUT] small dataset, batch size = 64, num epochs = 50, dropout model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/dropout.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-dropout-b64-small-e50 --num-epochs=50

## [NORM] small dataset, batch size = 64, num epochs = 50, norm model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/norm.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-norm-b64-small-e50 --num-epochs=50

## [LIN4] small dataset, batch size = 64, num epochs = 50, lin4 model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/lin4.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-lin4-b64-small-e50 --num-epochs=50

## [NORM-DROPOUT] small dataset, batch size = 64, num epochs = 50, norm with dropout model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/norm.dropout.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-norm.dropout-b64-small-e50 --num-epochs=50

## [LIN4-DROPOUT] small dataset, batch size = 64, num epochs = 50, lin4 with dropout model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/lin4.dropout.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-lin4.dropout-b64-small-e50 --num-epochs=50

## [NORM-LIN4] small dataset, batch size = 64, num epochs = 50, norm with lin4 model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/norm.lin4.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-norm.lin4-b64-small-e50 --num-epochs=50

## [NORM-LIN4-DROPOUT] small dataset, batch size = 64, num epochs = 50, norm-lin4-dropout model
python3 run_train_seqlab.py -t training/config/seqlab/non-overlap.b64.small.json -m models/config/seqlab/norm.lin4.dropout.json --device=cuda:0 --device-list=0,1,2,3 --run-name=seqlab-norm.lin4.dropout-b64-small-e50 --num-epochs=50

# gene sequential labelling training command
## [LSTM] gene sequential labelling with small dataset, num epochs = 50, base model
python3 run_train_genlab.py -t training/config/genlab/non-overlap.b64.small.json -m models/config/genlab/base.json --device=cuda:0 --device-list=0,1,2,3 --run-name=genlab-base-b64-small-e50 --num-epochs=50

## [GRU] gene sequential labelling with small dataset, num epochs = 50, gru model
python3 run_train_genlab.py -t training/config/genlab/non-overlap.b64.small.json -m models/config/genlab/gru.json --device=cuda:0 --device-list=0,1,2,3 --run-name=genlab-gru-b64-small-e50 --num-epochs=50

# multitask learning command
## run mtl training for three batch sizes: 64, 128, 256
python3 run_train_mtl.py -t training/config/mtl/mtl.balanced.b256.json -m models/config/mtl/base.json --device=cuda:0 --device-list=0,1,2,3 --run-name=mtl-base-b64-e50,mtl-base-b128-e50,mtl-base-b256-e50 --batch-sizes=64,128,256 --num-epochs=50