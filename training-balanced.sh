clear &&
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.eps1e-6.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss &&
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr2e-4.eps1e-6.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss &&
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr1e-5.eps1e-6.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss && 
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr2e-5.eps1e-6.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss && 
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr3e-5.eps1e-6.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss && 
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr5e-5.eps1e-6.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss &&
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss &&
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr2e-4.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss &&
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr1e-5.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss && 
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr2e-5.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss && 
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr3e-5.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss && 
python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr5e-5.json -m models/config/seqlab/ -c freeze.base,base -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss

