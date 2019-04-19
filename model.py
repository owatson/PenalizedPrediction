
import stats
import datetime
import numpy as np
from scipy.spatial.distance import cdist
import joblib

def full_model(df, val='POTENCY', outdir='results/', wins=True):
    fc_rf_hdr = {}
    fc_lr_hdr = {}
    ctr_hdr = {}  # this tells you the mean distance...
    fpa = np.asarray([x for x in df.fps.values])
    bins = [0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9]

    #bins = [0., 0.4, ]
    vals = df[val].values
    if wins:
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        vals = (vals - mean_val) / std_val
        pass

    assert(~np.isnan(mean_val))
    assert(~np.isnan(std_val))

    md = stats.get_model_dict()

    for i in range(len(bins)):
        fc_rf_hdr[i] = []
        fc_lr_hdr[i] = []
        ctr_hdr[i] = []
        pass

    for i in range(df.shape[0]):

        if i < 10:
            print ('Doing', i)
            print(datetime.datetime.now().time())
        elif i < 100:
            if (i % 10) == 0:
                print ('Doing', i)
                print(datetime.datetime.now().time())
        else:
            if (i % 100) == 0:
                print ('Doing', i)
                print(datetime.datetime.now().time())
                pass
            pass

        my_fp = np.asarray([df.fps.values[i], ])
        dists = cdist(my_fp, fpa, metric='jaccard')[0]
        dists[i] -= 2  # to exclude from anything (we're going to use >)

        for j in range(len(bins)):
            bstart = bins[j]

            gd = (dists >= bstart)

            prd_i_rf = md['rf']['m'](**md['rf']['kw']).fit(fpa[gd], vals[gd]).predict(my_fp)

            prd_i_rdg = md['ridge']['m'](**md['ridge']['kw']).fit(fpa[gd], vals[gd]).predict(my_fp)

            fc_rf_hdr[j].append(prd_i_rf[0])
            fc_lr_hdr[j].append(prd_i_rdg[0])

            ctr_hdr[j].append(np.mean(dists[gd]))
            pass
        pass

    joblib.dump((mean_val, std_val, ctr_hdr, fc_lr_hdr, fc_rf_hdr), outdir + val + '.results')