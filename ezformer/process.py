import numpy as np
import pandas as pd
import anndata as ad


#Re-load saved ref/var scores and process them
def process_predict_results(
        offsets, 
        targets_df,
        edits_df,
        experiment_prefix,
        model_type='enformer-tf',
        center_on_tss=True, re_center=False, score_rc=True,
        verbose=True):
    """
    Load saved prediction results and process them for downstream analysis.

    offsets: the list of offsets to load from those used for runing enformer like model
    target_df: the dataframe containing the meta data used in the enformer tracks
    edits_df: the dataframe containing the edits
    experiment_prefix: the prefix to save the results
    model_type: the model type used for prediction. Default is 'enformer-tf'
    center_on_tss: whether to center on TSS. Default is True
    re_center: whether to recenter the edits. Default is False
    score_rc: score reverse complement as another data augmentation. Default is True
    verbose: whether to print verbose output. Default is True
    """

    fold_index = [0]

    all_ref_scores = []
    all_var_scores = []

    tss_str = '_centered_on_tss' if center_on_tss else ''
    recentered_str = '_recentered' if re_center else ''

    #Loop over offsets
    for offset in offsets :
        #Load scores
        ref_scores = np.load(experiment_prefix + '_ref_scores_offset_' + str(offset) + tss_str + recentered_str + '.npy')
        if model_type == 'enformer-tf':
            all_ref_scores.append(ref_scores[:, fold_index, :])
        elif model_type == 'performer':
            all_ref_scores.append(ref_scores)
        #Load scores
        var_scores = np.load(experiment_prefix + '_var_scores_offset_' + str(offset) + tss_str + recentered_str + '.npy')
        if model_type == 'enformer-tf':
            all_var_scores.append(var_scores[:, fold_index, :])
        elif model_type == 'performer':
            all_var_scores.append(var_scores)
        
        if score_rc :
            ref_scores_rc = np.load(experiment_prefix + '_ref_scores_rc_offset_' + str(offset) + tss_str + recentered_str + '.npy')
            if model_type == 'enformer-tf':
                all_ref_scores.append(ref_scores_rc[:, fold_index, :])
            elif model_type == 'performer':
                all_ref_scores.append(ref_scores_rc)
            
            var_scores_rc = np.load(experiment_prefix + '_var_scores_rc_offset_' + str(offset) + tss_str + recentered_str + '.npy')
            if model_type == 'enformer-tf':
                all_var_scores.append(var_scores_rc[:, fold_index, :])
            elif model_type == 'performer':
                all_var_scores.append(var_scores_rc)
    
    #Aggregate scores over folds, offsets and reverse-complement ensemble
    if model_type == 'enformer-tf':
        ref_scores = np.mean(np.concatenate(all_ref_scores, axis=1).astype('float32'), axis=1)
        var_scores = np.mean(np.concatenate(all_var_scores, axis=1).astype('float32'), axis=1)
    elif model_type == 'performer':
        ref_scores = np.mean(all_ref_scores, axis=0).astype('float32').reshape(-1, 1)
        var_scores = np.mean(all_var_scores, axis=0).astype('float32').reshape(-1, 1)

    #Compute log2 fold-change scores: var_scores vs. ref_scores    

    scores = np.log2(var_scores / ref_scores)

    if verbose: print("scores.shape = " + str(scores.shape))

    #Cache final predicted scores (averaged across ensemble)

    adata = ad.AnnData(
        X=scores.T,
        obs=targets_df,
        var=edits_df
    )

    adata.layers['log2_fold_change'] = adata.X
    adata.layers['pct_change'] = 100. * (2**adata.X - 1.)

    return adata
