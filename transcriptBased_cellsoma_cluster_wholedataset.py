# parallized across the full dataset. 
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


# Get the list of donors 
ROOT_DATA_PATH = r'/mnt/xuisebnas2/Lab/Bereket_SEA_AD_analysis/SEA_AD_dataset'
SAVE_ROOT_PATH = r'/mnt/xuisebnas2/Lab/Bereket_SEA_AD_analysis/analysis_result'

def getSomaTranscript_segm(donorID):

    # First load all the detected transcripts for the donor 
    detected_transcript_fls = glob.glob(ROOT_DATA_PATH + os.sep + donorID + os.sep + '*' + os.sep + '*.csv')
    detected_transcript = []
    for fl in detected_transcript_fls:
        barcode = fl.split(os.sep)[-2]
        df = pd.read_csv(fl, index_col=0)
        df['Specimen Barcode'] = barcode
        detected_transcript.append(df)
    detected_transcript = pd.concat(detected_transcript,axis=0)

    # Load the adata object 
    adata = anndata.read_h5ad(r'/mnt/xuisebnas2/Lab/Bereket_SEA_AD_analysis/SEA_AD_dataset/combined_anndata_object/SEAAD_MTG_MERFISH.2024-12-11.h5ad')
    adata =  adata[~adata.obs.duplicated(['Specimen Barcode', 'Cell ID'])].copy() # remove the duplicated cells. 
    
    # Normalize the expression matrix -- log1p normalization 
    adata.layers['X_raw'] = adata.X.copy()
    total_count = adata.X.sum(axis=-1)
    sc.pp.normalize_total(adata, target_sum = np.median(total_count))
    sc.pp.log1p(adata)
    adata.layers['log1p_count'] = adata.X.copy()
    # cell type 
    celltype_list = list(set(adata.obs['Subclass']))


    all_genes = np.asarray(adata.var_names)
    keep_donor = adata.obs['Donor ID'] == donorID
    specimen_barcode = list(set(adata[keep_donor].obs['Specimen Barcode']))
    for specimen_ in specimen_barcode:
        keep_sample = adata.obs['Specimen Barcode'] == specimen_
        subadata = adata[keep_donor & keep_sample].copy()
        ds = detected_transcript[detected_transcript['Specimen Barcode'] == specimen_]
        if ds.empty:
            print(f'No transcripts for section: {donorID} {specimen_}')
            continue

        fov_groups = dict(tuple(ds.groupby('fov', sort=False)))
        cell_fovs = ds[ds['cell_id'] != -1].groupby('cell_id')['fov'].unique().to_dict()

        for e1 in celltype_list:
            # get the mean experssion across all cell-types 
            mean_all = np.mean(subadata[:,all_genes].X,axis=0)
            meanDH = np.mean(subadata[:,all_genes][subadata.obs['Subclass'] == e1].X,axis=0) - mean_all

            # sample cells across the layers 
            layers = set(subadata.obs['layer_annotation_CC']) # layer annotation using Celluar community analysis
            ibest_cells = []
            for ly in layers:
                iindex = np.flatnonzero((subadata.obs['layer_annotation_CC'] == ly) & (subadata.obs['Subclass'] == e1))
                # sameples max of 100 cells from each layer 
                rng = np.random.default_rng(seed=42) 
                NUM_CELLS = np.minimum(100,len(iindex)) 
                iindex = rng.choice(iindex, size=NUM_CELLS, replace=False)
                ibest_cells.append(subadata.obs['Cell ID'][iindex])
            ibest_cells = np.hstack(ibest_cells).tolist()

            for icellSEL in tqdm(ibest_cells, desc = f'{e1}:transcript density clustering'):
                SAVE_PATH = SAVE_ROOT_PATH + os.sep  + 'density_based_clustering_fulldataset' + os.sep + f'{donorID}' + os.sep + specimen_ + os.sep + f'{e1.replace("/","_")}_{icellSEL}.npz'
                os.makedirs(os.path.dirname(SAVE_PATH),exist_ok = True) # create the required folder 

                # inside the cell loop 
                fovs = cell_fovs.get(int(icellSEL),[])
                if len(fovs) == 0:
                    print(f'No transcripts - {e1}:{icellSEL}')
                    continue
                fov_detected_transcript = pd.concat([fov_groups[x] for x in fovs])
                # ID assigned to cells 
                icell = fov_detected_transcript['cell_id'].values
                genes = fov_detected_transcript['gene'].values

                # count the number of genes assigned to teh selected cells assigned within the fov
                gns_,cts = np.unique(genes[icell == int(icellSEL)], return_counts = True)
                dict_cts = dict(zip(gns_,cts/np.sum(cts)))

                # count ambient transcripts, transcripts not assigned to any cell (i.e. non-soma RNA)
                gns_,cts = np.unique(genes[icell == -1],return_counts = True)
                dict_cts_bk = dict(zip(gns_,cts/np.sum(cts)))
                # calculate teh enrichment of each genes, i.e. fold change with respect to the non-soma transcripts
                enrich = np.array([dict_cts[gn]/(dict_cts_bk.get(gn,0) + 1e-11) for gn in dict_cts])
                # keep only genes that show foldchange greater than 3 
                gns_kp = np.asarray(list(dict_cts.keys()))[enrich > 3]
                # also make sure the mean expression is greater than 0 
                gns_kp = np.intersect1d(gns_kp,all_genes[meanDH > 0])
                kp = np.isin(genes,gns_kp)
                # the spatial XYZ postion of the transcript spots
                Xall = fov_detected_transcript[['global_x','global_y','global_z']].values * [1,1,1.5]# this is already in microns.
                                                                                                    # except for the z-location
                
                cell_or_unassigned = (icell == -1) | (icell == int(icellSEL))
                keep_for_dbscan = kp & cell_or_unassigned
                Xall_kp_df = fov_detected_transcript[keep_for_dbscan] #  keep transcripts that are not accessed to other cells except the current cell.
                Xall_kp = Xall[keep_for_dbscan]
                X_cell_kp = Xall[(icell==int(icellSEL)) & kp]

                # Find the nearest neighbor to each soma transcript of the selected cell 
                N = 5
                dd,ii = scipy.spatial.KDTree(X_cell_kp).query(X_cell_kp,N)
                dth = np.median(dd[:,-1]) # median distance of each trancript from the other transcript 

                # find the nearest neighbor to each non-soma transcript 
                X_bk = Xall[(icell == -1) & kp]
                dd,ii = scipy.spatial.KDTree(X_bk).query(X_bk,N)
                dth_bk = np.median(dd[:,-1])

                try:
                    # cluster the transcript based on their density 
                    from sklearn.cluster import DBSCAN
                    dbscan = DBSCAN(eps = (dth + dth_bk)/2, min_samples = N)
                    lab = dbscan.fit(Xall_kp).labels_
                except:
                    print(f'Failed at - {e1}:{icellSEL}')
                    continue 
                # get the number of genes with in each DBSCAN cluster label that is associated within the selected cell
                
                # get the DBSCAN cluster label that contains most of the transcript using KDTree
                dd,ii = scipy.spatial.KDTree(Xall_kp).query(X_cell_kp)
                bd,cts = np.unique(lab[ii], return_counts = True)
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
                Xall_df = fov_detected_transcript[kpf]
                # save the result 
                np.savez_compressed(
                                    SAVE_PATH,
                                    X_somatic = X, # segmetnation based on somatic genes 
                                    genes_somatic_enr = genes[keep_for_dbscan][best == lab], # somatic enriched genes
                                    X_df = Xall_kp_df,  # transcript df
                                    Xall = Xall, # segmentation based on all genes 
                                    genes_all = genes_all, # all genes 
                                    Xall_df = Xall_df,
                                    pix_size = [1,1,1.5],

                                    allow_pickle = True
                                    )
            

if __name__ == '__main__':
    list_donors = [fl for fl in os.listdir(ROOT_DATA_PATH) if (fl.startswith('H')) and fl != 'H20.33.002']
    
    # donors to skip 
    with mp.Pool(processes=3) as pool:
        pool.map(getSomaTranscript_segm, list_donors)
