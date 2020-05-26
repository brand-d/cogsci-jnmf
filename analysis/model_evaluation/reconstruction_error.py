import numpy as np
import pandas as pd
import ccobra

def df_to_matrix(df):
    clean = df[["id", "task", "response"]]

    usr = list(clean["id"].unique())

    matrix = np.zeros((576, len(usr)))

    for _, row in clean.iterrows():
        usr_idx = usr.index(row["id"])
        syl_item = ccobra.Item(usr_idx, "syllogistic", row["task"], "single-choice", "", 0)
        syllog = ccobra.syllogistic.Syllogism(syl_item)
        enc_resp = syllog.encode_response(row["response"].split(";"))

        syl_idx = ccobra.syllogistic.SYLLOGISMS.index(syllog.encoded_task)
        resp_idx = ccobra.syllogistic.RESPONSES.index(enc_resp)
        comb_idx = syl_idx * 9 + resp_idx

        if matrix[comb_idx, usr_idx] != 0:
            print("Tried to write twice to field")
            exit()
        matrix[comb_idx, usr_idx] = 1

    return matrix
    
def split_by_property(data_df, property, use_median=True):
    if property not in data_df:
        print("Property '{}' not in dataset".format(property))
        exit()

    pivot = None
    if use_median:
        pivot = data_df[property].median()
    else:
        pivot = data_df[property].mean()

    above_df = data_df.loc[data_df[property] > pivot]
    below_df = data_df.loc[data_df[property] <= pivot]

    return above_df, below_df
    
def reconstruction_error(X, W, H):
    return (np.linalg.norm(X - W.dot(H.T), ord='fro') ** 2) / X.shape[1]
    
if __name__== "__main__":
    orig_data = pd.read_csv("../../data/exp_data_ccobra.csv")
    for property in ["conscientiousness", "openness"]:
        template_loading_string = "../../matrices/fit_result_{}_"
        
        # load matrices
        W_high = np.load((template_loading_string + "W_high.npy").format(property))
        W_low = np.load((template_loading_string + "W_low.npy").format(property))
        H_high = np.load((template_loading_string + "H_high.npy").format(property))
        H_low = np.load((template_loading_string + "H_low.npy").format(property))
        
        # get original matrices
        high_df, low_df = split_by_property(orig_data, property, use_median=True)
        X_high = df_to_matrix(high_df)
        X_low = df_to_matrix(low_df)
        
        print("-----------------------")
        print("Reconstruction errors for ", property)
        print("High (norm): ", reconstruction_error(X_high, W_high, H_high) / (X_high.shape[0] * X_high.shape[1]))
        print("Low  (norm): ", reconstruction_error(X_low, W_low, H_low) / (X_low.shape[0] * X_low.shape[1]))
        print()
        print("High (raw): ", reconstruction_error(X_high, W_high, H_high))
        print("Low  (raw): ", reconstruction_error(X_low, W_low, H_low))
        
        