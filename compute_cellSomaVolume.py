# run in rapid_scanpy enviroment 
import glob,os, numpy as np, pandas as pd 
from scipy.spatial import ConvexHull, cKDTree
import multiprocessing as mp 

ROOT = '/mnt/xuisebnas2/Lab/Bereket_SEA_AD_analysis/analysis_result/density_based_clustering_fulldataset_supertype'
ALPHA_UM = 4.0 # fixed for all cells so that we volume across cell types is comparable 
               # if we use adaptive cell alpha value it will partly reflect the transcript density of cell
               # Because alpha changes with each cell's point density, cells with fewer transcript get a different alpha
               # difference between supertypes could then partly reflect transcript counts rather than soma size. 
def alpha_volume(P,alpha=ALPHA_UM):
    import pyvista as pv 
    pv.vtk_verbosity('off')
    return pv.PolyData(P).delaunay_3d(alpha=alpha).volume

def section_volumes(fp):
    donor , specimen = fp.split(os.sep)[-2], os.path.basename(fp).removesuffix('.parquet')
    try:
        df = pd.read_parquet(fp, columns=['global_x','global_y','global_z', 'cell','supertype'],
                            filters = [('set','==','somatic_ds')])
    except Exception as e:
        print(f'FAILED {fp}: {e!r}', flush=True)
        return pd.DataFrame()
    
    rows = []
    for (cell,st),g in df.groupby(['cell','supertype'], observed=True): # NOTE: observed=True ensures we are keeping categoical values that have value.
        P = g[['global_x','global_y','global_z']].to_numpy(float)
        n_z = np.unique(P[:,2]).size # number of z-layers the cell spanned.
        r = dict(cell=cell, supertype=st, donor=donor, specimen=specimen,n_points=len(P),n_z=n_z,
                 hull_vol=np.nan, alpha_vol=np.nan,xy_area=np.nan)
        try:
            if len(P) >=4: r['xy_area'] = ConvexHull(P[:,:2]).volume # 2-D "volume" = area
            if len(P) >=12 and n_z >=2:
                r['hull_vol'] = ConvexHull(P).volume
                r['alpha_vol'] = alpha_volume(P)
        except Exception:
            pass
        rows.append(r)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    fls = sorted(glob.glob(f'{ROOT}/*/*.parquet'))
    with mp.Pool(16) as pool:
        vol = pd.concat(pool.map(section_volumes,fls),ignore_index=True)
    vol[['supertype','donor','specimen']] = vol[['supertype','donor','specimen']].astype('category') 
    out = os.path.join(os.path.dirname(ROOT), 'somaVolume_perSupertype', 'soma_volume_all.parquet')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vol.to_parquet(out, index=False)