import gc
import numpy as np
import torch
import enformer_pytorch
from time import time
from tangermeme.utils import one_hot_encode

#Define window to aggregate counts within
bin_size = 128
pad = 1088 ### ??? what is this
aggr_start_rel = -384       # 384 = 3 x 128bp bins
aggr_end_rel = 384          # 384 = 3 x 128bp bins
track_scales = 1.
clip_soft = None  ### i.e. they are not untransforming the soft clip
track_transforms = 1.


#Predict tracks
def predict_tracks(models, sequence_one_hot, n_folds=1):
    """Predict with enformer like model and extract the human tracks

    models: a list models probably for different folds for cross-validation
    sequence_one_hot: the one-hot-encoded sequence
    n_folds: the number of folds. Default is 1
    """

    predicted_tracks = []

    for fold_ix in range(n_folds) :
        yh = models[fold_ix].predict(sequence_one_hot[None, ...])['human'][:, None, ...].astype('float16')
        predicted_tracks.append(yh)

    predicted_tracks = np.concatenate(predicted_tracks, axis=1)

    return predicted_tracks


def predict_single_track_scalar(models, sequence_one_hot, n_folds=1):
    """Predict with enformer like model on a single track (pre-selected in the model)

    models: a list models probably for different folds for cross-validation
    sequence_one_hot: the one-hot-encoded sequence
    n_folds: the number of folds. Default is 1

    return: the predicted scalar value
    """

    predictions = []

    for fold_ix in range(n_folds) :
        yh = models[fold_ix].predict(sequence_one_hot).astype('float16')
        yh = yh[:,yh.shape[1]//2,:] #keep value at center of sequence. The sequence axis is removed
        predictions.append(yh)

    predictions = np.concatenate(predictions, axis=1)

    return predictions


#Function to undo transforms
def _undo_transforms(y_wt, y_mut, track_scales, clip_soft, track_transforms) :
    """
    probably undo the transforms applied in enformer!
    """

    # undo scale
    y_wt /= track_scales
    y_mut /= track_scales

    # undo soft_clip
    if clip_soft is not None :
        y_wt_unclipped = (y_wt - clip_soft)**2 + clip_soft
        y_mut_unclipped = (y_mut - clip_soft)**2 + clip_soft

        unclip_mask_wt = (y_wt > clip_soft)
        unclip_mask_mut = (y_mut > clip_soft)

        y_wt[unclip_mask_wt] = y_wt_unclipped[unclip_mask_wt]
        y_mut[unclip_mask_mut] = y_mut_unclipped[unclip_mask_mut]

    # undo sqrt
    y_wt = y_wt**(1. / track_transforms)
    y_mut = y_mut**(1. / track_transforms)
    
    return y_wt, y_mut


#Function to run enformer like model on a given offset and TSS position
def predict_single_offset_tss(
        models, 
        offset,
        fasta_open,
        edits_df,
        tss_pos,
        experiment_prefix,
        model_type='enformer-tf',
        strand_pair_mask=None, strand_pair_new=None,
        re_center=False, center_on_tss=True, score_rc=True, 
        seq_len = 393216,
        print_every = 16, collect_every = 32,
        n_folds=1,
        verbose=True):
    """
    Run enformer like model on a given offset and TSS position

    models: a list models probably for different folds for cross-validation
    offset: the offset to run enformer on, e.g. 0, -12, -24, -36, 12, 24, 36
    fasta_open: the opened fasta file
    edits_df: the dataframe containing the edits
    tss_pos: the TSS position to center on
    experiment_prefix: the prefix to save the results
    model_type: the model type. Default is 'enformer-tf'
    strand_pair_mask: the mask for the strand pairs (output of `preprocess_targets_df`)
    strand_pair_new: the new strand pairs (output of `preprocess_targets_df`)
    re_center: whether to recenter the edits. Default is False
    center_on_tss: whether to center on TSS. Default is True
    score_rc: score reverse complement as another data augmentation. Default is True
    seq_len: the sequence length for the given enformer model. Default is 393,216
    verbose: whether to print verbose output. Default is True
    """
   
    t0 = time()

    #Make predictions for all edits in dataframe (in chunks to manage memory)
    
    print("offset = " + str(offset))
    
    ref_scores = []
    ref_scores_rc = [] if score_rc else None
    
    var_scores = []
    var_scores_rc = [] if score_rc else None

    #Loop over dataframe
    for i, [_, row] in enumerate(edits_df.iterrows()) :

        if i % print_every == 0 :
            if verbose: print("i = " + str(i))

        insert_pos = row['position']

        chrom = row['chr']

        insert_seq = row['alt']
        insert_offset = 0
        delete_3prime = len(row['ref'])

        extra_padding = 1024

        # making sure TSS is captured (!) in both cases!
        if center_on_tss :
            start = tss_pos - seq_len // 2
            end = tss_pos + seq_len // 2

            mid_point = (insert_pos - start)

            aggr_start = tss_pos + aggr_start_rel
            aggr_end = tss_pos + aggr_end_rel
        else :
            start = insert_pos - seq_len // 2
            end = insert_pos + seq_len // 2

            mid_point = seq_len // 2

            aggr_start = insert_pos + aggr_start_rel
            aggr_end = insert_pos + aggr_end_rel

        aggr_start_bin = (aggr_start - start) // bin_size - pad
        aggr_end_bin = (aggr_end - start) // bin_size - pad

        #Get wildtype sequence
        seq_wt_orig = fasta_open.fetch(chrom, start - extra_padding, end + extra_padding).upper()

        rel_offset = len(insert_seq) // 2 if re_center else (len(insert_seq) - delete_3prime if center_on_tss and insert_pos + insert_offset < tss_pos else 0)

        #Insert edit
        seq_wt = seq_wt_orig[extra_padding+offset:][:seq_len]
        seq_mut = (seq_wt_orig[extra_padding+offset+rel_offset:extra_padding+mid_point+insert_offset] + insert_seq + seq_wt_orig[extra_padding+mid_point+insert_offset+delete_3prime:])[:seq_len]

        #Make predictions
        if model_type == 'enformer-tf' :
            #One-hot-encode
            sequence_one_hot_wt = np.array(one_hot_encode(seq_wt).permute(1,0).eq(1))
            sequence_one_hot_mut = np.array(one_hot_encode(seq_mut).permute(1,0).eq(1))

            y_wt = predict_tracks(
                models=models, 
                sequence_one_hot=sequence_one_hot_wt,
                n_folds=n_folds
            ).astype('float32')[:, :, aggr_start_bin:aggr_end_bin, :][..., strand_pair_mask]
            y_mut = predict_tracks(
                models=models, 
                sequence_one_hot=sequence_one_hot_mut,
                n_folds=n_folds
            ).astype('float32')[:, :, aggr_start_bin:aggr_end_bin, :][..., strand_pair_mask]
        
            #Undo transforms
            y_wt, y_mut = _undo_transforms(y_wt, y_mut, track_scales, clip_soft, track_transforms)
            
            score_ref = np.mean(y_wt, axis=2)
            score_var = np.mean(y_mut, axis=2)
        
        elif model_type == 'performer':
            #One-hot-encode
            sequence_one_hot_wt = one_hot_encode(seq_wt).permute(1,0).to(torch.float32)
            sequence_one_hot_mut = one_hot_encode(seq_mut).permute(1,0).to(torch.float32)

            y_wt = predict_single_track_scalar(
                models=models, 
                sequence_one_hot=sequence_one_hot_wt,
                n_folds=n_folds
            ).astype('float32')

            y_mut = predict_single_track_scalar(
                models=models, 
                sequence_one_hot=sequence_one_hot_mut,
                n_folds=n_folds
            ).astype('float32')
            
            score_ref = y_wt
            score_var = y_mut
        
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        
        ref_scores.append(score_ref)
        var_scores.append(score_var)
        
        y_wt_rc = None
        y_mut_rc = None

        #Optionally score rc
        if score_rc :
            if model_type == 'enformer-tf' :
                y_wt_rc = predict_tracks(
                    models=models, 
                    sequence_one_hot=sequence_one_hot_wt[::-1, ::-1],
                    n_folds=n_folds
                ).astype('float32')[:, :, ::-1, :][:, :, aggr_start_bin:aggr_end_bin, :][..., strand_pair_new][..., strand_pair_mask]
                y_mut_rc = predict_tracks(
                    models=models, 
                    sequence_one_hot=sequence_one_hot_mut[::-1, ::-1],
                    n_folds=n_folds
                ).astype('float32')[:, :, ::-1, :][:, :, aggr_start_bin:aggr_end_bin, :][..., strand_pair_new][..., strand_pair_mask]
            
                #Undo transforms
                y_wt_rc, y_mut_rc = _undo_transforms(y_wt_rc, y_mut_rc, track_scales, clip_soft, track_transforms)
            
                score_ref_rc = np.mean(y_wt_rc, axis=2)
                score_var_rc = np.mean(y_mut_rc, axis=2)
            
            elif model_type == 'performer':
                y_wt_rc = predict_single_track_scalar(
                    models=models, 
                    sequence_one_hot=sequence_one_hot_wt.flip(0,1),
                    n_folds=n_folds
                ).astype('float32')

                y_mut_rc = predict_single_track_scalar(
                    models=models, 
                    sequence_one_hot=sequence_one_hot_mut.flip(0,1),
                    n_folds=n_folds
                ).astype('float32')

                score_ref_rc = y_wt_rc
                score_var_rc = y_mut_rc
            
            else :
                raise ValueError(f"model_type {model_type} is not supported")

            ref_scores_rc.append(score_ref_rc)
            var_scores_rc.append(score_var_rc)

        #Collect garbage occasionally
        if i % collect_every == 0 :
            gc.collect()
    
    # convert lists to numpy arrays
    ref_scores = np.concatenate(ref_scores, axis=0).astype('float16')
    var_scores = np.concatenate(var_scores, axis=0).astype('float16')
    
    if score_rc:
        ref_scores_rc = np.concatenate(ref_scores_rc, axis=0).astype('float16')
        var_scores_rc = np.concatenate(var_scores_rc, axis=0).astype('float16')
    
    
    if verbose: print("ref_scores.shape = " + str(ref_scores.shape))
    if verbose: print("var_scores.shape = " + str(var_scores.shape))
    
    if score_rc:
        if verbose: print("ref_scores_rc.shape = " + str(ref_scores_rc.shape))
        if verbose: print("var_scores_rc.shape = " + str(var_scores_rc.shape))
    
    
    #Save predicted scores
    tss_str = '_centered_on_tss' if center_on_tss else ''
    recentered_str = '_recentered' if re_center else ''
    
    
    np.save(experiment_prefix + '_ref_scores_offset_' + str(offset) + tss_str + recentered_str + '.npy', ref_scores)
    np.save(experiment_prefix + '_var_scores_offset_' + str(offset) + tss_str + recentered_str + '.npy', var_scores)
    
    if score_rc:    
        np.save(experiment_prefix + '_ref_scores_rc_offset_' + str(offset) + tss_str + recentered_str + '.npy', ref_scores_rc)
        np.save(experiment_prefix + '_var_scores_rc_offset_' + str(offset) + tss_str + recentered_str + '.npy', var_scores_rc)

    print("done in %0.3fs" % (time() - t0))
