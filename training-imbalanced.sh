python3 run_train_mtl.py --training-config=training/config/mtl/mtl.imbalanced.b16.json --model-config=models/config/mtl/config_mtl.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-imbalanced.b16-base
python3 run_train_mtl.py --training-config=training/config/mtl/mtl.imbalanced.b32.json --model-config=models/config/mtl/config_mtl.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-imbalanced.b32-base
python3 run_train_mtl.py --training-config=training/config/mtl/mtl.imbalanced.b64.json --model-config=models/config/mtl/config_mtl.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-imbalanced.b64-base
python3 run_train_mtl.py --training-config=training/config/mtl/mtl.imbalanced.b128.json --model-config=models/config/mtl/config_mtl.json --device=cuda:0 --device-list=0,1,2,3,4,5,6,7 --run-name=mtl-imbalanced.b128-base