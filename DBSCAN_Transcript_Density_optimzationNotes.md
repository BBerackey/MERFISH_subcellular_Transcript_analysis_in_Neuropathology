Current implementation of tranascript denisty based clustering.

`
from tqdm import tqdm 
SAVE_ROOT_PATH = r'/mnt/xuisebnas2/Lab/Bereket_SEA_AD_analysis/analysis_result'
celltype_list = ['Sst','Pvalb','Sst Chodl','L2/3 IT','L5 IT']
all_genes = np.asarray(adata.var_names)
keep_donor = adata.obs['Donor ID'] == donorID
specimen_barcode = list(set(adata[keep_donor].obs['Specimen Barcode']))
for specimen_ in specimen_barcode:
    keep_sample = adata.obs['Specimen Barcode'] == specimen_
    subadata = adata[keep_donor & keep_sample].copy()
    for e1 in celltype_list:
        # get the mean experssion across all cell-types 
        mean_all = np.mean(subadata[:,all_genes].X,axis=0)
        meanDH = np.mean(subadata[:,all_genes][subadata.obs['Subclass'] == e1].X,axis=0) - mean_all

        # Get cells that show the mean gene expression 
        iindex = np.where(subadata.obs['Subclass'] == e1)[0]
        best_cor = np.argsort([np.corrcoef(meanDH,e)[0,1]  for e in subadata.X[iindex] - mean_all])[::-1]
        ibest_cells = subadata.obs['Cell ID'][iindex[best_cor]].values # cell-ids sorted based on correlation 
        # loop through at most top 500 cells.
        NUM_CELLS = np.minimum(100,len(iindex))

        for icellSEL in tqdm(ibest_cells[:NUM_CELLS], desc = f'{e1}:transcript density clustering'):
            SAVE_PATH = SAVE_ROOT_PATH + os.sep  + 'density_based_clustering' + os.sep + donorID + os.sep + specimen_ + os.sep + f'{e1}_{icellSEL}.npz'
            os.makedirs(os.path.dirname(SAVE_PATH),exist_ok = True) # create the required folder 
            # identify the fov within which the cells are located and restrict the detected transcripts to the FOV
            keep_transcripts = (detect_transcript['cell_id'] == icellSEL) & (detect_transcript['Specimen Barcode'] == specimen_)
            fov = detect_transcript.loc[keep_transcripts,'fov'].unique()
            # subset the transcript csv to transcripts within FOV for that specific donor and specimen barcode 
            keep_transcripts = (detect_transcript['fov'].isin(fov)) & (detect_transcript['Specimen Barcode'] == specimen_)
            fov_detected_transcript = detect_transcript.loc[keep_transcripts]

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
            Xall = fov_detected_transcript[['global_x','global_y','global_z']].values  * [1,1,1.5]# this is already in microns.
                                                                                                  # except for the z-location
            
            Xall_kp_df = fov_detected_transcript[kp]
            Xall_kp = Xall[kp]
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
            ulab = np.unique(lab)
            mean_cell = [np.mean(icell[kp][lab==ilab] == int(icellSEL)) for ilab in ulab]
            best = ulab[np.argmax(mean_cell)] # get the DBSCAN cluster label that contains most of the transcript 
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
                                genes_somatic_enr = genes[kp], # somatic enriched genes
                                X_df = Xall_kp_df,  # transcript df
                                Xall = Xall, # segmentation based on all genes 
                                genes_all = genes_all, # all genes 
                                Xall_df = Xall_df,
                                pix_size = [1,1,1.5],

                                allow_pickle = True
                                )

    break
            
`

Major issues:

1. **eps selection rule:** `eps = (dth + dth_bk)/2` is dominated by the background term. dth is about 1.7 um  and dth_bk is about 15 um, -> eps comes out at 8-13 um. which is about the radis of a soma or larger. This is not necesarly a probelm. As we want to capture the full size of the cell. we do not want it to be restricted to the segmentation assigned by cellpose.
2. **Depth dependence:** at 25% depth, the current method's median volume rises 1.86-fold (IQR 0.90–3.79). This matters if RNA quality differs across AD stages, because lower depth could look like larger cells.
3. **Cell selection is biased:** choosing the top 100 by correlation mostly selects high-count cells (Spearman /rho = 0.71-0.89 with spot count). This may exclude degenerating cells with atypical expression, which are essentially for comparing AD pathology. Therefore, sample randomly or stratify by depth and layer. 

