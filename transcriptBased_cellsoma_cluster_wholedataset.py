# parallized across the full dataset. 
# run in rapid-scanpy env 
import numpy as np
import pandas as pd
import glob
import scipy 
from tqdm import tqdm 
import anndata
import scanpy as sc 
import os
from tqdm import tqdm 
import sklearn
import multiprocessing as mp
import gc 
import time 

# Get the list of donors 
ROOT_DATA_PATH = r'/mnt/xuisebnas2/Lab/Bereket_SEA_AD_analysis/SEA_AD_dataset'
SAVE_ROOT_PATH = r'/mnt/xuisebnas2/Lab/Bereket_SEA_AD_analysis/analysis_result'
H5AD_PATH = r'/mnt/xuisebnas2/Lab/Bereket_SEA_AD_analysis/SEA_AD_dataset/combined_anndata_object/SEAAD_MTG_MERFISH.2024-12-11.h5ad'
layer_annot_PATH = r'/mnt/xuisebnas2/Lab/Bereket_SEA_AD_analysis/analysis_result/cellular_comunity_cluster/GMM_based_layerAnnot.csv'

def process_sections(task):
    """
    Perform the cell soma clustering 
    """
    # 1. load in the detected transcript 
    donorID, speciemenID = task['donor'],task['specimen']
    fl = ROOT_DATA_PATH + os.sep + donorID + os.sep + speciemenID + os.sep + r'cellpose-detected_transcripts.csv'
    # make sure the file exists 
    if not os.path.exists(fl):
        print(f'Detected transcript for donor:{donorID} - specimen:{speciemenID}  missing')
        return 
    ds = pd.read_csv(fl, usecols=['global_x','global_y','global_z','fov','gene','cell_id'],
                 dtype={'global_x':'float32','global_y':'float32','global_z':'float32',
                        'fov':'int32','gene':'category','cell_id':'int64'})
    # identify the fov groups 
    fov_groups = dict(tuple(ds.groupby('fov', sort = False)))
    cell_fovs = ds[ds['cell_id'] != -1].groupby('cell_id')['fov'].unique().to_dict()
    # for each cell type compute the transcript based somatic clustering 
    ds_result = [] # full dataframe, subset of detected transcript 
    for e1 in task['cells']:
        ibest_cells = task['cells'][e1] # cells sampled across layers 
        ibest_genes = task['genes'][e1] # genes that are differentially expressed in the cell 
        for icellSEL in tqdm(ibest_cells, desc = f'{e1}:transcript density clustering',disable=True):
            try:
                # get the detected transcripts within the fov associated with the cell 
                fovs = cell_fovs.get(int(icellSEL),[])
                if len(fovs) == 0:
                    print(f'No transcript -- {e1}:{icellSEL}')
                    continue
                fov_ds = pd.concat([fov_groups[x] for x in fovs])
                # ID assigned to genes and cells 
                icell = fov_ds['cell_id'].values
                genes = fov_ds['gene'].values

                # count the number of genes assigned to the selected cells within the fov 
                gns_,cts = np.unique(genes[icell == int(icellSEL)], return_counts = True)
                dict_cts = dict(zip(gns_,cts/np.sum(cts)))

                # get the count of ambient transcripts, transcripts not assigned to any cell (i.e. non-soma RNA)
                gns_,cts = np.unique(genes[icell == -1],return_counts = True)
                dict_cts_bk = dict(zip(gns_,cts/np.sum(cts)))
                # calculate the enrichment of each genes, i.e. fold-change with respect to the non-somatic transcripts
                enrich = np.array([dict_cts[gn]/(dict_cts_bk.get(gn,0) + 1e-11) for gn in dict_cts])
                # keep only genes that show foldchange greater than 3 
                gns_kp = np.asarray(list(dict_cts.keys()))[enrich > 3]
                # also make sure the genes have above baseline expression in the cell type 
                gns_kp = np.intersect1d(gns_kp,ibest_genes)
                kp = np.isin(genes,gns_kp) # mask for the transcripts based on gene expression 
                # the spatial XYZ postion of the transcript spots
                Xall = fov_ds[['global_x','global_y','global_z']].values * [1,1,1.5]# this is already in microns.

                # consider only transcripts that are assigned to the cell or ambinent, this minimzes merging of close by cells
                cell_or_unassigned = (icell == -1) | (icell == int(icellSEL))
                keep_for_dbscan = kp & cell_or_unassigned
                Xall_kp_df = fov_ds[keep_for_dbscan] #  keep transcripts that are not accessed to other cells except the current cell.
                Xall_kp = Xall[keep_for_dbscan]
                X_cell_kp = Xall[(icell==int(icellSEL)) & kp]

                # Find the nearest neighbor to each soma transcript of the selected cells 
                N = 5
                dd,ii = scipy.spatial.KDTree(X_cell_kp).query(X_cell_kp,N)
                dth = np.median(dd[:,-1]) # median distance of each cellular transcript from the other celluar transcripts 

                # find the nearest neighbor to each non-soma transcript 
                X_bk = Xall[(icell == -1) & kp]
                dd,ii = scipy.spatial.KDTree(X_bk).query(X_bk,N)
                dth_bk = np.median(dd[:,-1])
                # if the are fewer than N=5 points. KDTRee.query returns inf distances, so dth becomes inf,
                # and DBSCAN(eps=inf) raises InvalidParameterError. check this 
                if (len(X_cell_kp) <= 5) | (len(X_bk) <= 5):
                    print(f'Too few transcripts ({len(X_cell_kp)}) - {e1}:{icellSEL}', flush=True)
                    continue      
                # perform DBSCAN clustering 
                # cluster the transcript based on their density 
                from sklearn.cluster import DBSCAN
                dbscan = DBSCAN(eps = (dth + dth_bk)/2, min_samples = N)
                lab = dbscan.fit(Xall_kp).labels_
                # get the DBSCAN cluster label that contains most of the transcript using KDTree
                dd,ii = scipy.spatial.KDTree(Xall_kp).query(X_cell_kp)
                bd,cts = np.unique(lab[ii], return_counts = True)
                # exclude the background(noisy) cluster label (i.e. -1 ) from being selected 
                bd,cts = bd[bd != -1],cts[bd != -1]
                if len(bd) == 0: # skip if all clusters are noisy.
                    continue
                best = bd[np.argmax(cts)]

                X = Xall_kp[best==lab] # get all the transcripts with the most dense DBSCAN cluster, 
                                        # which is most likly the **soma transcripts**
                                        # NOTE: this is based on transcript enriched in the soma. 
                Xall_kp_df = Xall_kp_df[best==lab]
                
                # include using all transcripts 
                dd,ii = scipy.spatial.KDTree(X).query(Xall)
                kpf  = dd<2
                Xall = Xall[kpf,:]
                genes_all = genes[kpf]
                Xall_df = fov_ds[kpf]
                # -----------------------
                # prepare as a dataframe 
                ds_result.append(Xall_kp_df.assign(cell=int(icellSEL), supertype=e1,set='somatic_ds'))
                ds_result.append(Xall_df.assign(cell=int(icellSEL), supertype=e1,set='all_ds'))
                
            except Exception as e:
                print(f'FAILED {donorID}/{speciemenID} {e1}:{icellSEL} -> {e!r}', flush=True)  
                continue  
    # save result as a parquet file 
    if not ds_result:
        print(f'No results for {donorID}/{speciemenID}', flush=True)
        return
    out = f'{SAVE_ROOT_PATH}/density_based_clustering_fulldataset_supertype/{donorID}/{speciemenID}.parquet'
    os.makedirs(os.path.dirname(out),exist_ok=True)
    import pdb
    pdb.set_trace()
    res = pd.concat(ds_result, ignore_index = True)
    res['global_z'] *=1.5 # multiply by 1.5 to change to microns. 
    res[['supertype','set']] = res[['supertype','set']].astype('category')
    res.to_parquet(out, index = False)
    print(f'saved {out}: {res["cell"].nunique()} cells, {len(res)} rows', flush=True)
        
def prepare_tasks():
    """
    Prepare input require to process each tissue section form each sample. 
    """
    # Load the adata object 
    adata = anndata.read_h5ad(H5AD_PATH)
    # add the laminar annotation 
    GMM_layer_annot = pd.read_csv(layer_annot_PATH, index_col=0)
    GMM_layer_annot.index = GMM_layer_annot.index.astype(str)
    adata.obs = adata.obs.join(GMM_layer_annot['layer_gmm'])
    adata =  adata[~adata.obs.duplicated(['Specimen Barcode', 'Cell ID'])].copy() # remove the duplicated cells. 
    
    # Normalize the expression matrix -- log1p normalization 
    total_count = adata.X.sum(axis=-1)
    sc.pp.normalize_total(adata, target_sum = np.median(total_count))
    sc.pp.log1p(adata)
    # cell type 
    celltype_list = list(set(adata.obs['Supertype']))
    all_genes = np.asarray(adata.var_names)
    
    # Get the combination of speciemen and donor-id 
    specimen_to_donor = adata.obs[['Donor ID','Specimen Barcode']].set_index('Specimen Barcode').to_dict()['Donor ID']
    # for each specimen of each donor prepare the cellID - sampled from each layer 
    tasks = []
    for specimen_, donor_ in tqdm(specimen_to_donor.items(), desc = 'Preparing tasks'):
        keep_cells = (adata.obs['Donor ID'] == donor_) & (adata.obs['Specimen Barcode'] == specimen_)
        sub_obs = adata[keep_cells].obs
        layers = set(sub_obs['layer_gmm']).difference(['unassigned']) # layer annotations obtained from GMM prediction
        Xs = np.asarray(adata.X[keep_cells.values])
        mean_all = Xs.mean(0)
        ibest_cells = {}
        ibest_genes = {}
        for e1 in celltype_list:
            # sample cells from each layer 
            ibest_cells_ = []
            for lyr in layers:
                iindex = np.flatnonzero((sub_obs['layer_gmm'] ==lyr).values & (sub_obs['Supertype'] ==e1).values)
                # sample max of 100 cells from each layer 
                rng = np.random.default_rng(seed=42)
                NUM_CELLS = np.minimum(100,len(iindex))
                iindex = rng.choice(iindex, size = NUM_CELLS, replace = False)
                if len(iindex) == 0: # if no cells are sampled from a layer
                    continue
                ibest_cells_.append(sub_obs['Cell ID'].iloc[iindex])
            if not ibest_cells_:
                continue
            ibest_cells[e1] = np.hstack(ibest_cells_).tolist()
            # get the list of differentially expressed genes in the cell-type 
            meanDH = Xs[(sub_obs['Supertype'] == e1).values].mean(0) - mean_all
            ibest_genes[e1] = all_genes[meanDH > 0]
        tasks.append({'donor':donor_,
                      'specimen':specimen_,
                      'cells': ibest_cells,
                      'genes':ibest_genes
                    })
    del adata 
    gc.collect()
    return tasks

if __name__ == '__main__':
    
    tasks = prepare_tasks()
    # save the tasks, 
    pd.to_pickle(tasks,SAVE_ROOT_PATH + os.sep + 'process_tasks.pkl')
    print('Running tasks...')
    with mp.Pool(processes=8,maxtasksperchild=1) as pool:
        pool.map(process_sections, tasks, chunksize=1)
    print('DONE!')