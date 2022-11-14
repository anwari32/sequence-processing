import os
from Bio.Emboss.Applications import WaterCommandline
import pandas as pd
from tqdm import tqdm

def parse_score_line(line):
    words = line.split(":")
    score = float(words[1])
    return score

def parse_identity_or_similarity_line(line):
    val = line.split(":")[1].strip()
    vals = val.split(" ")
    vals = vals[0].split("/")
    upper_val = float(vals[0])
    lower_val = float(vals[1])
    score = round(upper_val/lower_val, 3)
    return score

if __name__ == "__main__":
    data_dir = os.path.join("error-analysis", "data-comparison")
    test_data = os.path.join(data_dir, "test_data.csv")
    validation_data = os.path.join(data_dir, "validation_data.csv")
    training_data = os.path.join(data_dir, "training_data.csv")
    water_output_dir = os.path.join("error-analysis", "alignment", "water")
    if not os.path.exists(water_output_dir):
        os.makedirs(water_output_dir, exist_ok=True)

    for p in [test_data, validation_data, training_data]:
        if os.path.exists(p):
            print(f"path found {p}")
        else:
            raise FileNotFoundError(f"path not found at {p}")


    test_df = pd.read_csv(test_data)
    training_df = pd.read_csv(training_data)
    test_similarity_scores = []
    test_identity_scores = []
    test_water_scores = []
    for i, r in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Running Water Algorithm"):
        test_seq = r["sequence"]
        identity_scores = []
        similarity_scores = []
        water_scores = []
        for j, s in training_df.iterrows():
            training_seq = s["sequence"]
            cline = WaterCommandline(gapopen=10, gapextend=0.5)
            cline.asequence = f"asis:{test_seq}"
            cline.bsequence = f"asis:{training_seq}"
            temp_output = os.path.join(water_output_dir, f"water[{i}][{j}].txt")
            cline.outfile = temp_output
            cline()
            output = open(temp_output, "r").readlines()
            identity_line = parse_identity_or_similarity_line(output[23])
            similarity_line = parse_identity_or_similarity_line(output[24])
            score_line = parse_identity_or_similarity_line(output[26])
        
        test_identity_scores.append(identity_scores)
        test_similarity_scores.append(similarity_scores)
        test_water_scores.append(water_scores)
            