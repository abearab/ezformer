import numpy as np
import pandas as pd


#Preprocess targets dataframe (i.e. the "enformer_targets_human" file from the enfoemer repo)
def preprocess_targets_df(dataset_path='datasets/', write=False):

    targets_file = f'{dataset_path}/enformer_targets_human.txt'
    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')

    #Get strands

    targets_df['strand'] = '.'
    targets_df['strand_pair'] = targets_df.index.values

    #Get strand mask
    index_orig = targets_df.index.values.tolist()
    index_orig_to_new = { i_orig : i_new for i_new, i_orig in enumerate(index_orig) }

    strand_pair_orig = targets_df['strand_pair'].values.tolist()
    strand_pair_new = np.array([index_orig_to_new[i_orig] for i_orig in strand_pair_orig], dtype='int32')

    strand_pair_mask = np.array(np.nonzero(np.array((targets_df['strand'] != '-').values))[0], dtype='int32')

    #Get forward-stranded dataframe
    targets_df_strand = targets_df.iloc[strand_pair_mask].copy()
    
    if write:
        targets_df_strand.to_csv(f'{dataset_path}/enformer_targets_human_strand.txt', index=True, sep='\t')

    return targets_df, targets_df_strand, strand_pair_mask, strand_pair_new
