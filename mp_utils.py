from __future__ import division, print_function

import numpy as np
import pandas as pd
import os
import glob
from .utils import get_canonical_smile, get_fp, get_qed, get_crippen
from rdkit import Chem
import joblib
from scipy.spatial.distance import cdist
import pdb
# Various utilities for dealing with the molport data.

raw_files = glob.glob('/Users/oliverwatson/evartech/molport/iissc*')
std_files = glob.glob('/Users/oliverwatson/evartech/molport/standardn_*')

def generate_data():
    """
    This adds additional data to the Molport files.
    :return:
    """
    for (i, fn) in enumerate(raw_files):
        sfn = '/Users/oliverwatson/evartech/molport/standardn_' + str(i)
        if os.path.isfile(sfn):
            continue
        print('Doing', fn)
        df1 = pd.read_csv(fn, low_memory=False, sep='\t')
        df1['SMILESIS'] = pd.Series([get_canonical_smile(x) for x in df1.SMILES_CANONICAL.values])
        # Construct fingerprints from ISIDRO smiles (so we don't eg have salts)
        fps = [get_fp(x) for x in df1.SMILESIS.values]
        df1['fps'] = pd.Series([np.asarray(fp).astype(bool) for fp in fps], index=df1.index)
        hfps = [get_fp(x, hashed=True) for x in df1.SMILESIS.values]
        df1['hfps'] = pd.Series([np.asarray(fp) for fp in hfps], index=df1.index)
        df1.to_pickle(sfn)
        pass
    return



def intersect_molport(df, name='dataframe'):
    """

    :param df:
    :param name:
    :return:
    """
    chmbl_smiles = set(df.SMILESIS.values)
    intersections = []
    for fn in std_files:
        dft = pd.read_pickle(fn)
        mol_smiles = set(dft.SMILESIS.values)
        intersections.append(mol_smiles.intersection(chmbl_smiles))
        pass
    all_ints = set.union(*intersections)
    print('%d in %s' % (len(chmbl_smiles), name))
    print('%d in full intersection' % len(all_ints))
    return all_ints


def get_mp_price(tdf, ix):
    """

    :param tdf:
    :param ix:
    :return:
    """
    v = tdf.PRICERANGE_1MG.values[ix]
    if v == 'nan':
        return 'nan'
    if '-' in v or '<' in v or '>' in v:
        return v + '/1MG'
    v = tdf.PRICERANGE_5MG.values[ix]
    if '-' in v or '<' in v or '>' in v:
        return v + '/5MG'
    v = tdf.PRICERANGE_50MG.values[ix]
    return v + '/50MG'

def get_mp_labels(tdf, idcs, potency, crippen, tox, qed, md):
    """

    :param tdf:
    :param idcs:
    :param potency:
    :param crippen:
    :param tox:
    :param qed:
    :return:
    """

    mp_labels = {'SMILES' : [], 'LABELS' : [], 'POTENCY' : potency[idcs], 'LOGP' : crippen[idcs],
                 'PRICES' : [], 'NEIGHBOUR_SMILES' : [], 'DISTANCES' : [],
                 }

    for ix in idcs:
        mp_labels['SMILES'].append(tdf.SMILESIS.values[ix])
        mp_labels['DISTANCES'].append('TD={:.3f}'.format(tdf.min_dist.values[ix]))
        try:
            px = get_mp_price(tdf, ix)
        except TypeError:
            px = 'nan'
        mp_labels['PRICES'].append(px)
        label = 'P={:.2f} L={:.2f} T={:.2f} Q={:.2f}'.format(
            potency[ix], crippen[ix], tox[ix], qed[ix])
        mp_labels['LABELS'].append(label)
        mp_labels['NEIGHBOUR_SMILES'].append(tdf.min_smile[ix])

    return mp_labels

def get_results(fn, full_models, num=10, log_p_thresh=4, tox_thresh=25,
                qed_thresh=0.5, rf_sc_tx=0.5, drugs=[], drug_dist=0.6):
    """
    - going to return four types of results, keyed by tag.

    POT:    (potency) the {num} most potent.
    LIP:    (lipophilicity) the {num} that score highest on potency - logP
    LIPO:   potency - logP/12
    BEST_O: (best Ollie)  the most potent subject to:
           - logP < logPthresh
           - tox < 0.25
    BEST_Q: (best QED) - as above but with QED > 0.5
    """

    df1 = pd.read_pickle(fn)
    preds = np.vstack(df1['fps'])


    if 'crippen' not in df1:
        df1['qed'] = pd.Series([get_qed(Chem.MolFromSmiles(sm)) for sm in df1.SMILESIS.values], index=df1.index)
        df1['crippen'] = pd.Series([get_crippen(Chem.MolFromSmiles(sm)) for sm in df1.SMILESIS.values],
                                   index=df1.index)
        df1.to_pickle(fn)
        pass
    crippen = df1.crippen.values
    qed = df1.qed.values
    md = df1.min_dist.values
    # Potency
    #potency = ( 1 -rf_sc_pot) * full_models['power']['ridge'].predict(preds)
    #potency  += rf_sc_pot * full_models['power']['rf'].predict(preds)

    # New potency model...
    #pms = full_models['power']['dist'](df1.min_dist.values)
    #pdb.set_trace()
    #potency = 3.5 * (1-pms) + pms * full_models['power']['rf'].predict(preds)

    fa = full_models['frac_act'](df1.min_dist.values)
    rfb = full_models['rfb'](df1.min_dist.values)
    inactive_level = full_models['inactive_level']
    active_level = full_models['active_level']

    potency = fa * (rfb * full_models['power']['rf'].predict(preds) + (1-rfb)*active_level) + (1-fa)*inactive_level

    gd = ~np.isnan(crippen)
    tox = 100 * np.ones_like(potency)


    preds = np.asarray([x for x in df1['hfps'].values[gd]])
    preds = np.hstack((preds, crippen[gd ,None]))

    tox[gd] = ( 1 -rf_sc_tx) * full_models['tox']['ridge'].predict(preds)
    tox[gd] += rf_sc_tx * full_models['tox']['rf'].predict(preds)

    # highest potency...
    best_pot = np.argsort(-potency)[:num]

    lip = potency - crippen
    lip[np.isnan(lip)] = -100

    lipo = potency - crippen/12
    lipo[np.isnan(lipo)] = -100

    best_lip = np.argsort(-lip)[:num]
    best_lipo = np.argsort(-lipo)[:num]

    # best ollie
    ollie = potency.copy()
    ollie[np.isnan(crippen)] = 0
    ollie[crippen > log_p_thresh] = 0
    ollie[np.isnan(tox)] = 0
    ollie[tox > tox_thresh] = 0

    best_ollie = np.argsort(-ollie)[:num]

    ollie[qed < qed_thresh] = 0
    best_ollie_q = np.argsort(-ollie)[:num]

    for drug in drugs:
        drug_fp = get_fp(drug)
        dists = cdist(np.asarray([drug_fp,]), np.vstack(df1['fps']), metric='jaccard')
        ollie[dists[0] < drug_dist] = 0
        pass

    best_ollie_d = np.argsort(-ollie)[:num]

    return {'POT': get_mp_labels(df1, best_pot, potency, crippen, tox, qed, md),
            'LIPO': get_mp_labels(df1, best_lipo, potency, crippen, tox, qed, md),
            'BEST_O': get_mp_labels(df1, best_ollie, potency, crippen, tox, qed, md),
            'BEST_Q': get_mp_labels(df1, best_ollie_q, potency, crippen, tox, qed, md),
            'BEST_D': get_mp_labels(df1, best_ollie_d, potency, crippen, tox, qed, md),
            }

def get_best(full_models, num=50, drugs=[], drug_dist=0.6):
    """

    :param num:
    :return:
    """
    hdr = []
    count = 0
    for fn in std_files:
        count += 1
        print(count)
        hdr.append(get_results(fn, full_models, num=num,
                               drugs=drugs, drug_dist=drug_dist))
        pass
    joblib.dump(hdr, 'molport_best')


def summarize(typ='POT', num=20, draw=True, plabel=False,
              check_new=None, neighbours=False, min_dist=0.1):
    smiles = []
    min_smiles = []
    labels = []
    values = []
    prices = []
    min_dists = []

    best = joblib.load('molport_best')
    for b in best:
        smiles += list(b[typ]['SMILES'])
        labels += list(b[typ]['LABELS'])
        pot = list(b[typ]['POTENCY'])
        prices += list(b[typ]['PRICES'])
        min_smiles += list(b[typ]['NEIGHBOUR_SMILES'])
        min_dists += list(b[typ]['DISTANCES'])

        if typ == 'LIP':
            logp = b[typ]['LOGP']
            for (i, p) in enumerate(pot):
                values.append(p - logp[i])
            pass
        elif typ == 'LIPO':
            logp = b[typ]['LOGP']
            for (i, p) in enumerate(pot):
                values.append(p - logp[i]/12)
            pass
        else:
            values += pot
        pass

    # here we need to clean up duplicate smiles but using Chem <=> smiles (this gets rid of
    # stereoisomers.   this is clunky and should be re-written.

    best_pot_holder = {}
    for (i, smile) in enumerate(smiles):
        cmbl_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        if cmbl_smile in best_pot_holder:
            if best_pot_holder[cmbl_smile]['value'] < values[i]:
                best_pot_holder[cmbl_smile]['value'] = values[i]
                best_pot_holder[cmbl_smile]['smile'] = smiles[i]
                best_pot_holder[cmbl_smile]['label'] = labels[i]
                best_pot_holder[cmbl_smile]['price'] = prices[i]
                best_pot_holder[cmbl_smile]['min_smile'] = min_smiles[i]
                best_pot_holder[cmbl_smile]['min_dist'] = min_dists[i]
            pass
        else:
            best_pot_holder[cmbl_smile] = {'label' : labels[i], 'smile' : smiles[i],
                                           'price' : prices[i], 'min_smile' : min_smiles[i],
                                           'min_dist' : min_dists[i],  'value' : values[i],
                                           }
        pass

    smiles = []
    min_smiles = []
    labels = []
    values = []
    prices = []
    min_dists = []

    for hdr in best_pot_holder.values():
        smiles.append(hdr['smile'])
        min_smiles.append(hdr['min_smile'])
        values.append(hdr['value'])
        prices.append(hdr['price'])
        min_dists.append(hdr['min_dist'])
        labels.append(hdr['label'])
        pass



    if check_new is not None:
        for i in range(len(values)):
            cd = cdist(np.asarray([get_fp(smiles[i]), ]), np.asarray([v for v in check_new.fps.values]),
                       metric='jaccard')[0]
            smallest = cd[np.argsort(cd)][0]
            if smallest < min_dist:
                values[i] = 0
                pass
            pass
        pass

    idcs = np.argsort(-1 * np.asarray(values))[:num]
    if draw:
        if neighbours:
            lbls = [min_dists[x] for x in idcs]
            return Chem.Draw.MolsToGridImage([Chem.MolFromSmiles(min_smiles[x]) for x in idcs],
                                        legends=lbls)
        if plabel:
            lbls = [prices[x] for x in idcs]
        else:
            lbls = [labels[x] for x in idcs]
        return Chem.Draw.MolsToGridImage([Chem.MolFromSmiles(smiles[x]) for x in idcs],
                                        legends=lbls)

    else:
        if neighbours:
            return [min_smiles[x] for x in idcs], [min_dists[x] for x in idcs]
        return [smiles[x] for x in idcs], [labels[x] for x in idcs]
