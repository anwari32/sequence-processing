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


python run_sequence_labelling.py -t training/config/seqlab/ss-only.01.lr5e-5.json -m models/config/seqlab/ -c base.lin1 -d cuda:4 --device-list=4,5,6,7 --use-weighted-loss


python create_baseline_data_bundle_from_index.py -i index/gene_index.01_test.csv -s data/gene_dir -d workspace/baseline/basic -c 150 -k 50 &&
python create_baseline_data_bundle_from_index.py -i index/gene_index.01_train_validation.csv -s data/gene_dir -d workspace/baseline/basic -c 150 -k 50 &&
python split_train_validation.py -s workspace/baseline/basic/gene_index.01_train_validation_ss_all_pos.csv -d workspace/baseline/basic

python create_baseline_data_bundle_from_index.py -i index/gene_index.01_test.csv -s data/gene_dir -d workspace/baseline/kmer -c 150 -k 1 --kmer &&
python create_baseline_data_bundle_from_index.py -i index/gene_index.01_train_validation.csv -s data/gene_dir -d workspace/baseline/kmer -c 150 -k 1 --kmer &&
python split_train_validation.py -s workspace/baseline/kmer/gene_index.01_train_validation_ss_all_pos.csv -d workspace/baseline/kmer
