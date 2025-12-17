from pathlib import Path
from collections import Counter
from pprint import pprint
import numpy as np

if __name__=="__main__":
    data_dir = Path("/usr/people/mdodson/sport-lis-smvi/tmp")

    smvi_jon_files = [
        "smvi_la_20230909_jon-mod_county0-10-fd.npy",
        "smvi_la_20230909_jon-mod_county0-40-fd.npy",
        "smvi_la_20230909_jon-mod_county0-100-fd.npy",
        "smvi_la_20230909_jon-mod_county0-200-fd.npy",
        ]
    smvi_jon = np.stack(list(map(
        np.load, map(data_dir.joinpath, smvi_jon_files)
        )), axis=-1)[np.newaxis]

    #smvi_jon[np.isnan(smvi_jon)] = -1

    smvi_mitch = np.load(data_dir.joinpath(
        "smvi_la_20230909_mitchell.npy"))

    '''
    (jtm_vals,),(jtm_counts,) = np.unique(
            smvi_mitch[smvi_jon==1], return_counts=True))
    (mtj_vals,),(mtj_counts,) = np.unique(
            smvi_jon[smvi_mitch==1], return_counts=True))
    print(f"unique in jon array:",jtm_vals)
    print(f"unique in mitch array:",mtj_vals)
    print(f"SMVI in both:",jtm_counts[np.argmax(jtm_vals)])
    print(f"SMVI in mitch, not jon:",jtm_counts[np.argmax(jtm_vals)])
    '''

    unqm,unqj = map(np.unique, [smvi_mitch,smvi_jon])
    coords = np.stack(np.meshgrid(
        unqm, unqj, indexing="ij"
        ), axis=-1).reshape(-1,2).tolist()

    combos = {}
    for mv,jv in coords:
        mask = (smvi_mitch == mv) & (smvi_jon == jv)
        combos[(mv,jv)] = np.count_nonzero(mask)

    print(f"unique in jon array:", np.unique(unqj))
    print(f"unique in mitch array:", np.unique(unqm))
    pprint(combos)
