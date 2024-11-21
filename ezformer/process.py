import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


#Re-load saved ref/var scores and process them
def process_enformer_results(
        experiment_prefix, 
        offsets, 
        center_on_tss=True, re_center=False, score_rc=True,
        write_to_file=False,
        verbose=True):

    fold_index = [0]

    all_ref_scores = []
    all_var_scores = []

    tss_str = '_centered_on_tss' if center_on_tss else ''
    recentered_str = '_recentered' if re_center else ''

    #Loop over offsets
    for offset in offsets :
        #Load scores
        ref_scores = np.load(experiment_prefix + '_ref_scores_offset_' + str(offset) + tss_str + recentered_str + '.npy')
        all_ref_scores.append(ref_scores[:, fold_index, :])
        #Load scores
        var_scores = np.load(experiment_prefix + '_var_scores_offset_' + str(offset) + tss_str + recentered_str + '.npy')
        all_var_scores.append(var_scores[:, fold_index, :])
        
        if score_rc :
            ref_scores_rc = np.load(experiment_prefix + '_ref_scores_rc_offset_' + str(offset) + tss_str + recentered_str + '.npy')
            all_ref_scores.append(ref_scores_rc[:, fold_index, :])
            
            var_scores_rc = np.load(experiment_prefix + '_var_scores_rc_offset_' + str(offset) + tss_str + recentered_str + '.npy')
            all_var_scores.append(var_scores_rc[:, fold_index, :])

    #Aggregate scores over folds, offsets and reverse-complement ensemble
    ref_scores = np.mean(np.concatenate(all_ref_scores, axis=1).astype('float32'), axis=1)
    var_scores = np.mean(np.concatenate(all_var_scores, axis=1).astype('float32'), axis=1)

    #Compute log2 fold-change scores: var_scores vs. ref_scores    

    scores = np.log2(var_scores / ref_scores)

    if verbose: print("scores.shape = " + str(scores.shape))

    #Cache final predicted scores (averaged across ensemble)

    if write_to_file:
        np.save(experiment_prefix + '_final_scores' + tss_str + recentered_str + '.npy', scores)

    return scores


#Function to plot predictions
def plot_predictions(df, df_col, scores, targets_df, df_col_pred):
    target_index = targets_df.index
    
    score_log2 = np.mean(scores[:, target_index], axis=1)
    pct_change = 100. * (2**score_log2 - 1.)
    
    df[df_col_pred] = pct_change

    df_sub = df.loc[~(df[df_col].isnull() | df[df_col_pred].isnull())]

    y_pred = np.array(df_sub[df_col_pred].values)
    y_true = np.array(df_sub[df_col].values)

    rs = spearmanr(y_pred, y_true)[0]
    rp = pearsonr(y_pred, y_true)[0]

    f = plt.figure(figsize=(3, 3))

    plt.scatter(y_pred, y_true, s=4, color='black')
    
    plt.axhline(y=0, color='darkgreen', linewidth=1, linestyle='--')
    plt.axvline(x=0, color='darkgreen', linewidth=1, linestyle='--')
    
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.xlabel(df_col_pred, fontsize=8)
    plt.ylabel(df_col, fontsize=8)
    
    plt.title("Spearman r = " + str(round(rs, 3)) + "\n Pearson r = " + str(round(rp, 3)), fontsize=8)

    plt.tight_layout()

    plt.show()
