import tensorflow as tf
import pandas as pd
import numpy as np
import os
import wandb

if __name__ == "__main__":

    # static path and model configs.
    work_dir = os.path.join("run", "baseline", "kmer")
    model_dir = os.path.join(work_dir, "model")
    log_dir = os.path.join(work_dir, "log")

    bilstm_model = tf.keras.models.load_model(os.path.join(model_dir, "model_bilstm.h5"))
    bigru_model = tf.keras.models.load_model(os.path.join(model_dir, "model_bigru.h5"))

    data_dir = os.path.join("workspace", "seqlab-latest")
    test_data = os.path.join(data_dir, "gene_index.01_test_ss_all_pos.csv")

    from run_baseline_kmer import preprocessing
    X_test, Y_test = preprocessing(test_data)

    print(f"{X_test.shape}, {Y_test.shape}")
    
    # initialize wandb.
    run = wandb.init(
        project="prediction",
        entity="anwari32"
    )
    wandb.define_metric("prediction_step")
    wandb.define_metric("prediction/*", step_metric="prediction_step")
    logpath = os.path.join("prediction", "baseline-kmer")
    logpath = os.path.join(logpath, f"{run.id}.csv")
    if os.path.exists(logpath):
        os.remove(logpath)
    os.makedirs(os.path.dirname(logpath), exist_ok=True)

    for m in [("bilstm", bilstm_model), ("bigru", bigru_model)]:
        
        ypred = model.predict(X_test)

        model_name = m[0]
        model = m[1]

        y_pred = model.predict(X_test, Y_test)
        model.save(
            os.path.join(model_dir, f"model_{model_name}.h5")
        )
        hist_keys = train_history.history.keys()
        hist_data = {}
        for k in hist_keys:
            hist_data[k] = train_history.history[k]

        train_f1_score = []
        val_f1_score = []

        for p, r in zip(train_history.history["precision"], train_history.history["recall"]):
            train_f1_score.append(
                compute_f1_score(p, r)
            )

        for p, r in zip(train_history.history["val_precision"], train_history.history["val_recall"]):
            val_f1_score.append(
                compute_f1_score(p, r)
            )

        hist_data["f1_score"] = train_f1_score
        hist_data["val_f1_score"] = val_f1_score

        training_validation_result_df = pd.DataFrame(data=hist_data)
        training_validation_result_df.to_csv(
            os.path.join(log_dir, f"log.{model_name}.csv"), 
            index=False)
