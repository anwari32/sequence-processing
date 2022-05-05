# MTL
python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b16.json --model-config=models/config/mtl/config_mtl.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-balanced.b16-base
python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b32.json --model-config=models/config/mtl/config_mtl.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-balanced.b32-base
python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b64.json --model-config=models/config/mtl/config_mtl.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-balanced.b64-base
python3 run_train_mtl.py --training-config=training/config/mtl/mtl.balanced.b128.json --model-config=models/config/mtl/config_mtl.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-balanced.b128-base

# Seqlab
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/seqlab.training.non-overlap.b16.base.json --model-config=models/config/seqlab/config_seqlab.json
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/seqlab.training.non-overlap.b16.base.json --model-config=models/config/seqlab/config_seqlab_base1.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-base1-nonoverlap-b16
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/seqlab.training.non-overlap.b16.base.json --model-config=models/config/seqlab/config_seqlab_base2.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=seqlab-base2-nonoverlap-b16
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/seqlab.training.non-overlap.b32.base.json --model-config=models/config/seqlab/config_seqlab.json
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/seqlab.training.non-overlap.b64.base.json --model-config=models/config/seqlab/config_seqlab.json
python3 run_train_seqlab_gene.py --training-config=training/config/seqlab/by_genes/seqlab.training.non-overlap.b128.base.json --model-config=models/config/seqlab/config_seqlab.json