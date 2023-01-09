# utilities fot PCA analysis.

import pandas as pd
import matplotlib.pyplot as plt
from utils.seqlab import splice_site_ids, Index_Dictionary
from sklearn.decomposition import PCA


class PCA_Util:
    def __init__(self, dataframe: pd.DataFrame, feature_columns=[], n_components=2):
        self.pca = PCA(n_components=n_components)
        feature_frame = dataframe[feature_columns]
        self.data_shape = dataframe.shape
        self.feature_shape = feature_frame.shape
        feature_frame_std = (feature_frame - feature_frame.mean()) / feature_frame.std()
        principal_components = self.pca.fit_transform(feature_frame_std)
        principal_components_df = pd.DataFrame(data=principal_components, columns=[f"nf-{i}" for i in range(n_components)])

        self.principal_components = principal_components_df
        self.principal_shape = principal_components_df.shape
        self.principal_columns = principal_components_df.columns

        # merging principal components.
        for col in principal_components_df.columns:
            dataframe[col] = principal_components_df[col]
        
        self.dataframe = dataframe

    
    def export(self, dest_path):
        self.dataframe.to_csv(dest_path, index=False)
        

    def plot(self, dataframe):
        nrows = 2
        ncols = 3
        idx = 0
        figs, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=(8, 4))
        unique_label_ids = [0] + [7] + splice_site_ids
        for r in range(nrows):
            for c in range(ncols):
                i = unique_label_ids[idx]
                filtered_df = dataframe[dataframe["prediction_id"] == i]
                correct_df = dataframe[(dataframe["prediction_id"] == i) & (dataframe["target_id"] == i)]
                false_df = dataframe[(dataframe["prediction_id"] != i) & (dataframe["target_id"] == i)]
                correct_x = correct_df["nf-0"]
                correct_y = correct_df["nf-1"]
                false_x = false_df["nf-0"]
                false_y = false_df["nf-1"]
                axes[r][c].plot(
                    correct_x, 
                    correct_y,
                    '.',
                    color="green",
                    label=f"correct"
                )
                axes[r][c].plot(
                    false_x, 
                    false_y,
                    'x',
                    color="red",
                    label=f"incorrect"
                )
                axes[r][c].set_xlabel("PCA 1")
                axes[r][c].set_ylabel("PCA 2")
                axes[r][c].set_title(f"{Index_Dictionary[i]}")
                # axes[r][c].legend(loc="upper right")
                idx += 1

        plt.legend(loc="lower right")
        plt.show()


        