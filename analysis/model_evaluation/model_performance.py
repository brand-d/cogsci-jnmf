import numpy as np
import pandas as pd
import ccobra
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys

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

def get_response_vector(property, pattern):
    result = []
    prediction_matrix = pattern.reshape(64, 9)
    for i in range(64):
        result.append(ccobra.syllogistic.RESPONSES[prediction_matrix[i].argmax()])
    return result

def load_dataset(data_df, property):
    high, low = split_by_property(data_df, property)
    print("High: {}, Low: {}".format(len(high["id"].unique()), len(low["id"].unique())))
    return high, low

def evaluate(data, model, property="extraversion"):
    data_name, data_df = data
    model_name, model_pattern = model
    model_pred = get_response_vector(property, model_pattern)

    result = []
    for subj, subj_df in data_df.groupby("id"):
        for _, task in subj_df.iterrows():
            task_list = [x.split(";") for x in task["task"].split("/")]
            resp_list = task["response"].split(";")
            task_enc = ccobra.syllogistic.encode_task(task_list)
            resp_enc = ccobra.syllogistic.encode_response(resp_list, task_list)

            pred = model_pred[ccobra.syllogistic.SYLLOGISMS.index(task_enc)]
            hit = (resp_enc == pred)

            result.append({
                "data_name": data_name,
                "group": "{} model".format(model_name),
                "property": property,
                "subj": subj,
                "task": task_enc,
                "truth": resp_enc,
                "pred": pred,
                "hit": hit
            })
    return result

def get_model_performance(high_df, low_df, property, models):
    result = []
    # High group model
    result.extend(
        evaluate(("High {}".format(property), high_df), ("High", models[0])))
    result.extend(
        evaluate(("Low {}".format(property), low_df), ("High", models[0])))
    # Common model
    result.extend(
        evaluate(("High {}".format(property), high_df), ("Common", models[1])))
    result.extend(
        evaluate(("Low {}".format(property), low_df), ("Common", models[1])))
    # Low group model
    result.extend(
        evaluate(("High {}".format(property), high_df), ("Low", models[2])))
    result.extend(
        evaluate(("Low {}".format(property), low_df), ("Low", models[2])))
    result_df = pd.DataFrame(result)

    return result_df
    
def plot_performance(high_df, low_df, property, models, H_high, H_low):
    f = plt.figure(figsize=(6, 3))
    sns.set(palette="colorblind")
    sns.set_style("whitegrid", {'axes.grid' : True})

    top_ax = plt.gca()

    # Model performance
    result_df = get_model_performance(high_df, low_df, property, models)

    cp = sns.pointplot(
        x="data_name", y="hit", hue="group", data=result_df, ax=top_ax)
    cp.set(xlabel='', ylabel='Accuracy')
    top_ax.legend(
        title="", frameon=False, loc='upper center',
        ncol=3, bbox_to_anchor=(0.5, 1.15))
    top_ax.set_ylim(min(0.2, np.round(top_ax.get_ylim()[0], 1)),
                    np.round(top_ax.get_ylim()[1], 1))

    plt.tight_layout()

    plt.savefig('model_performance_{}.pdf'.format(property))
    plt.show()
    
if __name__== "__main__":
    properties = ["conscientiousness", "openness"]

    template_loading_string = "../../matrices/fit_result_{}_"

    for property in properties:
        print("property: {}".format(property))
    
        # load syllogistic dataset
        data_df = pd.read_csv("../../data/exp_data_ccobra.csv")
        high_df, low_df = load_dataset(data_df, property)

        # load matrices
        W_high = np.load((template_loading_string + "W_high.npy").format(property))
        W_low = np.load((template_loading_string + "W_low.npy").format(property))
        H_high = np.load((template_loading_string + "H_high.npy").format(property))
        H_low = np.load((template_loading_string + "H_low.npy").format(property))

        # extract patterns
        high_pattern = W_high[:,1]
        low_pattern = W_low[:,1]
        common_pattern = (W_high[:,0] + W_low[:,0]) / 2
        models = [high_pattern, common_pattern, low_pattern]

        # plotting
        plot_performance(high_df, low_df, property, models, H_high, H_low)
