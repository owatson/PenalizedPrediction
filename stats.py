import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model as LM
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV, BayesianRidge, ElasticNet, Lasso

from scipy.spatial.distance import pdist


model_dict = {'ridge' : {'m': Ridge, 'kw': {'fit_intercept': True, 'alpha': 0.1}},
              'rcv':  {'m': RidgeCV, 'kw': {'cv': 5}},
              'rf': {'m': RandomForestRegressor, 'kw': {'n_estimators': 100, 'n_jobs': 4, 'max_depth': 10}}}

fc_dict = {'tox' : {'base' : 'hfps', 'add' : ['crippen',], 'resp' : 'PCT_INHIB_HEPG2'},
           'power' : {'base' : 'fps', 'add' : [], 'resp' : 'POTENCY' },
           }


def fit_models(tox_df, pot_df):
    """
    Fits potency and toxicity models.
    :param tox_df: Dataframe used to get toxicity data
    :param pot_df: Dataframe used to get potency data
    :return: dictionary of models
    """
    mdls = {}

    for mdl_typ in fc_dict.keys():
        df = tox_df if mdl_typ == 'tox' else pot_df
        mdls[mdl_typ] = {}
        preds_base = df[fc_dict[mdl_typ]['base']]
        resp = df[fc_dict[mdl_typ]['resp']]
        gd = ~np.isnan(resp)
        for pred in fc_dict[mdl_typ]['add']:
            gd = gd & ~ np.isnan(df[pred].values)
            pass

        preds = np.asarray([x for x in preds_base.values[gd]])

        for pred in fc_dict[mdl_typ]['add']:
            preds = np.hstack((preds, df[pred].values[gd, None]))

        for typ in ['rf', 'ridge']:
            mdls[mdl_typ][typ] = model_dict[typ]['m'](**model_dict[typ]['kw']).fit(preds, resp[gd])
            pass
        pass

    return mdls


def get_model_dict():
    """
    Gets us the various machine learning models we're interested in, with their various parameters.
    :return:
    """
    return model_dict


def regress(response, pred_list, one=False, do_print=True):
    """
    Basic regression function
    :param response:
    :param pred_list:
    :param one:
    :param do_print:
    :return:
    """

    regr = LM.LinearRegression(fit_intercept=one)
    regr.fit(np.asarray(pred_list).swapaxes(0, 1), response)

    presp = regr.predict(np.asarray(pred_list).swapaxes(0, 1))
    r2 = r2_score(response, presp)
    if do_print:
        if one:
            rs = ('%6.4f ' * (len(pred_list) + 1)) % (tuple(regr.coef_) + (regr.intercept_,))
        else:
            rs = ('%6.4f ' * (len(pred_list)) % tuple(regr.coef_))
        print ('Coeffs:       ' + rs)
        print ('R-squared: {:9.4f}'.format(r2))
    else:
        return regr.coef_, r2


def run_suite(df, v, typ='df', adjust=None, label=None, add_pred=None):
    """

    :param df:
    :param v:
    :param typ:
    :param adjust:
    :param label:
    :param add_pred:
    :return:
    """
    if typ == 'df':
        print("Analyzing", v)
        pd = df[v].values
    elif typ == 'prec':  # precalculated
        print("Analyzing", label)
        pd = v
    else:
        raise ValueError

    gd = ~np.isnan(pd)
    if adjust is not None:
        gd = gd & ~np.isnan(adjust)
        pass

    if add_pred is not None:
        gd = gd & ~np.isnan(add_pred.sum(axis=1))
        pass

    adj = None if adjust is None else adjust[gd]

    for m in ['ridge', 'rf']:
        pred = np.asarray([x for x in df.fps[gd]])
        if add_pred is not None:
            pred = np.hstack((pred, add_pred[gd]))
        full_bootstrap(pred, np.asarray(pd[gd]), m, adjust=adj)
    print ('\nHashed...')
    for m in ['ridge', 'rf']:
        pred = np.asarray([x for x in df.hfps[gd]])
        if add_pred is not None:
            pred = np.hstack((pred, add_pred[gd]))
        full_bootstrap(pred, np.asarray(pd[gd]), m, adjust=adj)
    print('\n')


def full_bootstrap(preds, resps, method, num_runs=40, adjust=None):
    """

    :param preds:
    :param resps:
    :param method:
    :param num_runs:
    :param adjust:
    :return:
    """
    r_2s = []
    betas = []

    N = len(resps)
    for i in range(num_runs):
        idcs = np.random.choice(N, N)
        oos = np.delete(np.arange(N), idcs)

        mdl = model_dict[method]['m'](**model_dict[method]['kw'])
        mdl.fit(preds[idcs], resps[idcs])
        if adjust is None:
            beta, r2 = regress(resps[oos], [mdl.predict(preds[oos]), ], do_print=False, one=False)
        else:
            beta, r2 = regress(resps[oos] + adjust[oos],
                               [mdl.predict(preds[oos]) + adjust[oos], ], do_print=False, one=False)
        if beta < 0:
            r2 *= -1
            pass
        r_2s.append(r2)
        betas.append(beta)
        pass

    r_2s = np.asarray(r_2s)
    betas = np.asarray(betas)

    print ('Beta:         %9.3f [%8.3f %8.3f] %s' %
           (np.mean(betas), np.percentile(betas, 5), np.percentile(betas, 95), method))
    print ('R2 (signed):  %9.3f [%8.3f %8.3f] at 5pct conf' %
           (np.mean(r_2s), np.percentile(r_2s, 5), np.percentile(r_2s, 95)))


def binner(df, bin_label, sample_size=900, num_bins=5):
    """

    :param df:
    :param bin_label:
    :param sample_size:
    :param num_bins:
    :return:
    """
    bin_dict = {'dist_bin': [], 'mean': [], 'stdr': [], 'count': []}

    dfr = df[df[bin_label].notnull()]
    dfr = dfr[dfr.fps.notnull()]
    N = dfr.shape[0]

    bin_size = (N / num_bins)

    bin_buckets = (np.argsort(np.argsort(dfr[bin_label].values)) / bin_size).astype(int)

    sample = np.random.choice(N, size=sample_size, replace=False)

    # How many bins away are they?
    bin_dist = pdist(bin_buckets[sample, None], 'minkowski', 1)

    # Tanimoto distance on same sample
    t_dist = pdist(np.asarray([df.fps.values[x] for x in sample]), 'jaccard')

    for i in range(num_bins):
        vals = t_dist[bin_dist == i]
        # can be caused by a few bad fingerprints
        vals = vals[~np.isnan(vals)]
        bin_dict['dist_bin'].append(i)
        bin_dict['mean'].append(np.mean(vals))
        ct = len(vals)
        bin_dict['stdr'].append(np.std(vals) / np.sqrt(ct))
        bin_dict['count'].append(ct)
        pass

    return pd.DataFrame.from_dict(bin_dict)



