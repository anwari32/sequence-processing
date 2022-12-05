python run_pairwise_water.py -t "error-analysis\data-comparison\validation_data.csv" -b error-analysis\data-comparison\validation_data.csv -d "water-validation.csv" -n 45
python run_pairwise_water.py -t "prediction\dataframe-0.0=F1 Score-0.1.csv" -b error-analysis\data-comparison\validation_data.csv -d "water-0.0-0.1.csv" -n 45
python run_pairwise_water.py -t "prediction\dataframe-0.1=F1 Score-0.2.csv" -b error-analysis\data-comparison\validation_data.csv -d "water-0.1-0.2.csv" -n 45
python run_pairwise_water.py -t "prediction\dataframe-0.9=F1 Score-1.0.csv" -b error-analysis\data-comparison\validation_data.csv -d "water-0.9-1.0.csv" -n 45
python run_pairwise_water.py -t "prediction\dataframe-F1 Score=1.csv" -b error-analysis\data-comparison\validation_data.csv -d "water-1.0.csv" -n 45