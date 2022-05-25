python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b16.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b16-base --disable-wandb && python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b32.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b32-base --disable-wandb && python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b64.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b64-base --disable-wandb && python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b128.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b128-base --disable-wandb && python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b256.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b256-base --disable-wandb
python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b16.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b16-base --disable-wandb && python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b32.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b32-base --disable-wandb && python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b64.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b64-base --disable-wandb && python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b128.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b128-base --disable-wandb && 

python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b256.json --model-config=models/config/mtl/mtl.base0.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-b256-base --disable-wandb --num-epochs=50


python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/non-overlap.b1.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-b1-base
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/non-overlap.b16.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-b16-base
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/non-overlap.b32.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-b32-base --disable-wandb
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/non-overlap.b64.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-b64-base --disable-wandb
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/non-overlap.b128.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-b128-base --disable-wandb
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/non-overlap.b256.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-b256-base --disable-wandb

python3 run_train_seqlab.py --training-config=training/config/seqlab/by_sequence/non-overlap.b16.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-sequence-b16-base
python3 run_train_seqlab.py --training-config=training/config/seqlab/by_sequence/non-overlap.b32.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-sequence-b32-base
python3 run_train_seqlab.py --training-config=training/config/seqlab/by_sequence/non-overlap.b64.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-sequence-b64-base
python3 run_train_seqlab.py --training-config=training/config/seqlab/by_sequence/non-overlap.b256.json --model-config=models/config/seqlab/base.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-sequence-b256-base