
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from IPython.display import SVG
from rdkit.Chem.Draw import IPythonConsole
import joblib
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from IPython.display import HTML
#get_ipython().magic(u'pylab inline')


import pickle
with open('malaria_chembl.pickle', 'rb') as handle:
    b = pickle.load(handle)
a=[]
for i in b:
    a.append(i['assay_description'])
from collections import Counter
freq = Counter(a)


df_tcams = pd.read_pickle('chembl_tcams.pkl')


# The TCAMS data has two columns that will be of main interest to us:
# - pXC50_3D7: pIC50 values against the (3D7) strain of _Plasmodium Falciparum_.  Henceforth we will refer to this value (and the similar variable available in the Plasmodium dataset) as **Potency**.
# - PCT_INHIB_HEPG2: percent inhibition against HEPG2 cells, henceforth referred to as **Toxicity**.
# 
# The data extracted from ChEMBL only has antimalarial values (as mentioned earlier, we are interested in particular in potency).
# 
# Both datasets have ChEMBL compound IDs and SMILES associated to them.  To both datasets we associate canonical smiles as calculated by our own standardiser software, to ensure uniqueness.  From these canonical smiles we then separately calculate 1024 bit _binary fingerprints_ and 1024 bit _count fingerprints_.  These will be the predictors used in the subsequent analysis.  We also use these canonical smiles to look at the uniqueness and overlap between these two datasets, which we now describe.
# 
# #### Potency, selection, uniqueness and overlap in TCAMS and Plasmodium datasets
# 
# Following list of facts about TCAMS datsets is inferred from code run below:

# In[127]:


#from malaria import utils, stats
from utils import *
from stats import *
import stats
from standardiser import standardise

# These results indicate that the potency values in the TCAMS dataset extracted from ChEMBL and in the TCAMS data downloaded from the ChEMBL github repository are indeed the same, as indicated by the high correlation (i.e. 0.9998).

# In[145]:


#import utils, stats
#gd = ~np.isnan(df_tcams.pXC50_3D7.values)
#full_df = pd.DataFrame.from_dict({'SMILESIS' : list(TCAMS_from_ChEMBL_df.SMILESIS.values) + list(df_tcams.SMILESIS.values[gd]), 
#                                 'POTENCY'  : list(TCAMS_from_ChEMBL_df.VAL.values) + list(df_tcams.pXC50_3D7.values[gd]),
#                                 })
#full_df = full_df.groupby('SMILESIS').mean()
#fps = [utils.get_fp(x) for x in full_df.index]
#full_df['fps'] = pd.Series([np.asarray(fp).astype(bool) for fp in fps], index=full_df.index)
#hfps = [utils.get_fp(x, countfp=True) for x in full_df.index]
#full_df['hfps'] = pd.Series([np.asarray(fp) for fp in hfps], index=full_df.index)
#crippen = pd.Series([utils.get_crippen(Chem.MolFromSmiles(x)) for x in full_df.index], index=full_df.index)
#qed = pd.Series([utils.get_qed(Chem.MolFromSmiles(x)) for x in full_df.index], index=full_df.index)
#full_df['crippen'] = crippen
#full_df['qed'] = qed
#full_df.to_pickle('TCAMS_processed_potency_tox.pkl')
#full_df = pd.read_pickle('TCAMS_processed_potency.pkl')


# We now get inactives from ChEMBL from non-GSK screening assays

# In[123]:


full_df = pd.read_pickle('joined_chembl_data_w_inactives.pkl')


# In[130]:


print(len(full_df))
print(full_df.shape)


# Let's also add data on our standard anti-malarial drugs

# In[135]:


Artemisinin = 'CC1CCC2C(C(=O)OC3C24C1CCC(O3)(OO4)C)C'
Atovaquone  = 'C1CC(CCC1C2=CC=C(C=C2)Cl)C3=C(C4=CC=CC=C4C(=O)C3=O)O'
Chloroquine = 'CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl'
Doxycycline = 'CC1C2C(C3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O'
Mefloquine  = 'C1CCNC(C1)C(C2=CC(=NC3=C2C=CC=C3C(F)(F)F)C(F)(F)F)O'
Primaquine  = 'CC(CCCN)NC1=C2C(=CC(=C1)OC)C=CC=N2'
Piperaquine = 'C1CN(CCN1CCCN2CCN(CC2)C3=C4C=CC(=CC4=NC=C3)Cl)C5=C6C=CC(=CC6=NC=C5)Cl'
drugs = [Artemisinin, Atovaquone, Chloroquine, Doxycycline, Mefloquine, Primaquine, Piperaquine]
drugs = [standardise.run(drug) for drug in drugs]
drug_labels = ['Artemisinin', 'Atovaquone', 'Chloroquine', 
               'Doxycycline', 'Mefloquine', 'Primaquine', 'Piperaquine']


# Note - Doxycycline and Piperaquine aren't in our dataset - although something very similar to Piperaquine is (but nothing that close to Doxycycline). The others all are.

# In[147]:


from scipy.spatial.distance import pdist, cdist


# We should make some further comments about the potency data before going any further.
# 
# In the TCAMS dataset, there are no values for potency below ~ 5.3. This is because the TCAMS data only contains active molecules, as the compound structures for the inactive compounds tested by GSK were not released.
# 
# Therefore, it is clear that adding inactive data to the TCAMS dataset is going to be necessary, but given its biases towards active molecules anyway, our overall dataset will still be heavily biased, and in particularly very strongly biased towards predicting potency values that are unreasonably high. 
# 
# We checked for the best current known anti-malarial drugs (described in the next section).  All of these bar one (Doxycycline) are in the TCAMS dataset, **however**, **only two** have non-nan potency values (Chloroquine has two) - and the most potent known drug, Artemisinin has no potency value assigned.
# 
# Note that we do get potency values for all drugs in our merged dataset apart from Doxycycline and Piperaquine (and we do have a close match to Piperaquine).
# 
# Below we show the potency histogram of our full merged dataset:

# ### Figure 1.

# In[148]:


# In[138]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV

model_dict = {'ridge' : {'m': Ridge, 'kw': {'fit_intercept': True, 'alpha': 0.1}},
              'rcv':  {'m': RidgeCV, 'kw': {'cv': 5}},
              'rf': {'m': RandomForestRegressor, 'kw': {'n_estimators': 100, 'n_jobs': 4, 'max_depth': 10}}}

method = 'rf'
mdl = model_dict[method]['m'](**model_dict[method]['kw'])
target = full_df.POTENCY.values
gd = ~np.isnan(target)
preds = np.asarray([x for x in full_df.fps[gd]])



#df_tcams_nd.to_pickle('TCAMS_processed_tox.pkl')
df_tcams_nd = pd.read_pickle('TCAMS_processed_tox.pkl')

#
#
fpa = np.asarray([x for x in full_df.fps.values])
#
#fpa = np.asarray([x for x in full_df.fps.values])
#
#estimates = []
#weights = []
#
#for i in range(full_df.shape[0]):
#    v = [full_df.POTENCY.values[i]]
#    dists = cdist(np.asarray([full_df.fps.values[i],]), fpa)[0]
#    dists[i]+=1
#    gpot = full_df.POTENCY.values[dists == 0]
#    if len(gpot) == 0:
#        continue
#    gpot = np.concatenate((v, gpot))
#    estimates.append(np.std(gpot, ddof=1))
#    weights.append(gpot.shape[0]-1)
#    pass
#
#print('Standard deviation of potency values in our dataset')
#print(np.std(full_df.POTENCY.values, ddof=1))
#print('Standard deviation of compounds with identical fingerprints')
#print(np.average(estimates, weights=weights))
#print('Fraction of variance thus explained by fingerprints...')
#1-np.average(estimates, weights=weights)**2/np.std(full_df.POTENCY.values, ddof=1)**2


# This is a really important result.  The standard deviation of the potency values for malaria of two compounds with identical 1024-bit fingerprints is 0.185 (and we have 665 effective datapoints for this estimate, accounting for double-counting).  This means that the 1024-bit fingerprint captures 84% of the variance in potency (which is decent!)
# 
# What we actually want here is a plot showing the sensitivity of sigma to Tanimoto distance.  So let's construct that, and for toxicity, and plot them (and thus create figure 2 of the paper).
# 
# 
# First though - let's just show some examples for the reader of a couple of compounds that are different (and have different potency/toxicity values, but the same fingerprints, and also a couple of compounds that have opposite fingerprints...

# In[158]:


from sklearn.metrics.pairwise import pairwise_distances
#sum_fp = pd.Series([sum(x) for x in full_df.fps], index=full_df.index)
#full_df['good_fp'] = sum_fp
#full_dfg = full_df.loc[full_df['good_fp'] >0]
#v  = (pairwise_distances(np.asarray([x for x in full_dfg.fps.values]), metric='jaccard') == 0).nonzero()
#diffs = np.asarray([full_dfg.iloc[v[0][x]].POTENCY - full_dfg.iloc[v[1][x]].POTENCY for x in range(len(v[0]))])
#b = np.argmax(np.abs(diffs))
#a = [v[0][b], v[1][b]]
#labels = ['Potency %.2f' % (full_dfg.iloc[x].POTENCY) for x in a]
#
#
## In[159]:
#
#
#opposites  = (pairwise_distances(np.asarray([x for x in full_dfg.fps.values]), metric='jaccard') == 1).nonzero()
#ao = [opposites[0][0], opposites[1][0]]
#olabels = ['Potency %.2f' % (full_dfg.iloc[x].POTENCY) for x in ao]
#
#
## In[160]:
#
#
#my_smiles = [full_dfg.index[x] for x in [v[0][b], v[1][b]]]


# ### Figure 2.

# In[161]:


estimates_hdr = {}
weight_hdr = {}
ctr_hdr = {}

bins = [-0.001, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for i in range(len(bins)-1):
    estimates_hdr[i] = []
    weight_hdr[i] = []
    ctr_hdr[i] = []
    pass

for i in range(full_df.shape[0]):
    v = [full_df.POTENCY.values[i]]
    dists = cdist(np.asarray([full_df.fps.values[i],]), fpa, metric='jaccard')[0]
    dists[i]+=2 # to exclude from anything
    
    for j in range(len(bins)-1):
        bstart = bins[j]
        bend = bins[j+1]
        gd = (dists > bstart) & (dists <= bend)
        gpot = full_df.POTENCY.values[gd]
        if len(gpot) == 0:
            continue

        gpot = np.concatenate((v, gpot))
        estimates_hdr[j].append(np.std(gpot, ddof=1))
        weight_hdr[j].append(gpot.shape[0]-1)
        ctr_hdr[j].append(np.mean(dists[gd]))
        pass
    pass     

ctrs = np.asarray([np.average(ctr_hdr[i], weights=weight_hdr[i]) for i in range(len(bins)-1)])
sigmas = np.asarray([np.average(estimates_hdr[i], weights=weight_hdr[i]) for i in range(len(bins)-1)])

#stats.run_suite(df_tcams_nd, 'PCT_INHIB_HEPG2', add_pred=df_tcams_nd.crippen.values[:,None])

estimates_tox_hdr = {}
weight_tox_hdr = {}
ctr_tox_hdr = {}

bins = [-0.001, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
df_tox = df_tcams_nd.dropna(subset=['PCT_INHIB_HEPG2'])
fpat = np.asarray([x for x in df_tox.fps.values])


for i in range(len(bins)-1):
    estimates_tox_hdr[i] = []
    weight_tox_hdr[i] = []
    ctr_tox_hdr[i] = []
    pass

for i in range(df_tox.shape[0]):
    v = [df_tox.PCT_INHIB_HEPG2.values[i]]
    dists = cdist(np.asarray([df_tox.fps.values[i],]), fpat, metric='jaccard')[0]
    dists[i]+=2 # to exclude from anything
    
    for j in range(len(bins)-1):
        bstart = bins[j]
        bend = bins[j+1]
        gd = (dists > bstart) & (dists <= bend)
        gpot = df_tox.PCT_INHIB_HEPG2.values[gd]
        if len(gpot) == 0:
            continue

        gpot = np.concatenate((v, gpot))
        estimates_tox_hdr[j].append(np.std(gpot, ddof=1))
        weight_tox_hdr[j].append(gpot.shape[0]-1)
        ctr_tox_hdr[j].append(np.mean(dists[gd]))
        pass
    pass        


ctrst = np.asarray([np.average(ctr_tox_hdr[i], weights=weight_tox_hdr[i]) for i in range(len(bins)-1)])
sigmast = np.asarray([np.average(estimates_tox_hdr[i], weights=weight_tox_hdr[i]) for i in range(len(bins)-1)])

plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 20), dpi=400)
plt.subplot(211)
plt.plot(ctrs, sigmas, label='Data'); grid(True); xlabel('Tanimoto distance'); ylabel('Potency sigma')
legend(loc='best')
_ = title('Potency covariance as a function of Tanimoto distance')
plt.subplot(212)
plt.plot(ctrst, sigmast, label='Data'); grid(True); xlabel('Tanimoto distance'); ylabel('Toxicity sigma (%)')
legend(loc='best')
_ = title('Toxicity covariance as a function of Tanimoto distance')
savefig('figures/fig2_covariance.jpg')


# So now we understand how covariance works as a function of Tanimoto distance - let's see how our models perform when they are restricted to fitting on points above some minimum Tanimoto distance from the target point...

# In[176]:


fc_rf_hdr = {}
fc_lr_hdr = {}
ctr_hdr = {}  # this tells you the mean distance...

def rank(x):
    return np.argsort(np.argsort(x))/(len(x) + 0.) - 0.5

bins = [0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.9]


def run_big_fit():
    md = stats.get_model_dict()

    for i in range(len(bins)):
        fc_rf_hdr[i] = []
        fc_lr_hdr[i] = []
        ctr_hdr[i] = []
        pass
    

    for i in range(full_df.shape[0]):
    
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
            
        my_fp = np.asarray([full_df.fps.values[i],])
        dists = cdist(my_fp, fpa, metric='jaccard')[0]
        dists[i]-=2 # to exclude from anything (we're going to use >)
    
        for j in range(len(bins)):
            bstart = bins[j]

            gd = (dists >= bstart) 
    
            prd_i_rf = md['rf']['m'](**md['rf']['kw']).fit(fpa[gd], 
                    full_df.POTENCY.values[gd]).predict(my_fp)
        
            prd_i_rdg = md['ridge']['m'](**md['ridge']['kw']).fit(fpa[gd], 
                    full_df.POTENCY.values[gd]).predict(my_fp)
    
            fc_rf_hdr[j].append(prd_i_rf[0])
            fc_lr_hdr[j].append(prd_i_rdg[0])

            ctr_hdr[j].append(np.mean(dists[gd]))
            pass
        pass   
    return

run_big_fit()
joblib.dump(fc_rf_hdr, 'modelling/rf_fc_results')
joblib.dump(fc_lr_hdr, 'modelling/rdg_fc_results')
joblib.dump(ctr_hdr, 'modelling/ctr_results')
fc_rf_hdr = joblib.load('modelling/rf_fc_results')
fc_lr_hdr = joblib.load('modelling/rdg_fc_results')
ctr_hdr = joblib.load('modelling/ctr_results')


# In[175]:


import stats
result_hdr = {'rf' : [], 'rdg' : [], 'rfrk' : [], 'rdgrk' : [], 
             'rfb' : [], 'rdgb' : [], 'rfrkb' : [], 'rdgrkb' : [] 
             }#

print(fc_rf_hdr[i])
for i in range(len(bins)):
    rf_beta, rf_r2, rmse, r2_mean, rmse_mean = stats.regress(full_df.POTENCY.values, [fc_rf_hdr[i],], do_print=False, one=True)
    rdg_beta, rdg_r2, rmse, r2_mean, rmse_mean = stats.regress(full_df.POTENCY.values, [fc_lr_hdr[i],], do_print=False, one=True)
    rfr_beta, rfr_r2, rmse, r2_mean, rmse_mean = stats.regress(rank(full_df.POTENCY.values), [rank(fc_rf_hdr[i]),], 
                                     do_print=False)
    rdgr_beta, rdgr_r2, rmse, r2_mean, rmse_mean = stats.regress(rank(full_df.POTENCY.values), [rank(fc_lr_hdr[i]),], 
                                       do_print=False)   
    
    result_hdr['rf'].append(rf_r2)
    result_hdr['rfb'].append(rf_beta[0])
    result_hdr['rdg'].append(rdg_r2)
    result_hdr['rdgb'].append(rdg_beta[0])
    result_hdr['rfrk'].append(rfr_r2)
    result_hdr['rfrkb'].append(rfr_beta[0])
    result_hdr['rdgrk'].append(rdgr_r2)
    result_hdr['rdgrkb'].append(rdgr_beta[0])
       
result_hdr = joblib.load('results/summary_results')


# ### Figure 3.

# In[173]:


bins = [0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9]
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 20))
plt.subplot(221)
plt.plot(bins, result_hdr['rfb'], label='Random Forest Beta')
plt.plot(bins, result_hdr['rdgb'], label='Ridge regression Beta'); 
plt.plot(bins, np.zeros(len(bins)))
plt.xlabel('Minimum Tanimoto distance from training set')
plt.ylabel('Model strength')
plt.grid(True); plt.legend(loc='best')
plt.title('Potency model strength')
plt.subplot(222)
plt.plot(bins, result_hdr['rf'], label='Random Forest R2')
plt.plot(bins, result_hdr['rdg'], label='Ridge regression R2'); 
plt.plot(bins, np.zeros(len(bins)))
plt.xlabel('Minimum Tanimoto distance from training set')
plt.ylabel('R^2')
plt.ylim((-0.1, 0.4))
plt.grid(True); plt.legend(loc='best')
plt.title('Potency R2')

result_hdr = joblib.load('results_tox/summary_results')
plt.subplot(223)
plt.plot(bins, result_hdr['rfb'], label='Random Forest Beta')
plt.plot(bins, result_hdr['rdgb'], label='Ridge regression Beta'); 
plt.plot(bins, np.zeros(len(bins)))
plt.xlabel('Minimum Tanimoto distance from training set')
plt.ylabel('Model strength')
plt.grid(True); plt.legend(loc='best')
plt.title('Toxicity model strength')
plt.subplot(224)
plt.plot(bins, result_hdr['rf'], label='Random Forest R2')
plt.plot(bins, result_hdr['rdg'], label='Ridge regression R2'); 
plt.plot(bins, np.zeros(len(bins)))
plt.xlabel('Minimum Tanimoto distance from training set')
plt.ylabel('R^2')
plt.ylim((-0.1, 0.4))
plt.grid(True); plt.legend(loc='best')
_ = plt.title('Toxicity R2')
savefig('figures/fig3_str_r2.png')


# # 6. Dealing with the bias in the data set

# Unfortunately we need to deal with the bias in the data - hence we need to work out how to correct for it.  Here we run the analysis described in the paper to perform the bias correction.

# In[87]:


tc_dists = cdist(fpa, fpa, metric='jaccard') + np.diag(np.ones(len(fpa)) * np.nan)
import glob
std_files = glob.glob('/Users/oliverwatson/evartech/molport/standardn_*')
df0 = pd.read_pickle(std_files[0])
idcs = np.random.choice(np.arange(df0.shape[0]), 10000)
fpr = np.asarray([x for x in df0.iloc[idcs].fps.values])
r_dists = cdist(fpr, fpr, metric='jaccard') + np.diag(np.ones(len(fpr)) * np.nan)
tcr_dists = cdist(fpr, fpa, metric='jaccard')
mal_hist= np.histogram(tc_dists[~np.isnan(tc_dists)], density=True, bins=7)
mr_hist = np.histogram(tcr_dists[~np.isnan(tcr_dists)], density=True, bins=7)

# magic number - but justified by looking at the bottom plot below.
num_inactive_per_active = 100


# ### Figure 4.

# In[88]:


x_axis = 0.5*(mal_hist[1][1:] + mal_hist[1][:-1])
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 20))
plt.subplot(211)
plt.hist((tc_dists[~np.isnan(tc_dists)],tcr_dists[~np.isnan(tcr_dists)]), 
         bins=20, density=True, label=('Malaria self-distance', 'Random-malaria distance'))
#plt.hist(r_dists[~np.isnan(r_dists)], bins=20, label='Random distances')
plt.yscale('log')
plt.xlabel('Tanimoto distance')
plt.ylabel('Density')
plt.grid('True')
plt.legend(loc='best')
plt.title('Pairwise Tanimoto distance of non-identical compounds in malaria dataset vs random')
plt.subplot(212)
plt.plot(x_axis, mal_hist[0]/(num_inactive_per_active*mr_hist[0]))
plt.grid(True)
plt.xlabel('Tanimoto Distance')
plt.ylabel('Fraction of actives given 1% population active')
_ = plt.title('Estimate of fraction of actives as function of distance from an active')
savefig('figures/fig4_bias_correction.png')


# # 7. Using our models
# 
# Now we can actually create a proper extrapolative model to use in predicting on general compounds...

# In[89]:


from functools import partial
frac_active = partial(np.interp, xp=x_axis, fp=(mal_hist[0]/(num_inactive_per_active*mr_hist[0])).clip(0., 1.))

# magic number - but with some emprical support in the paper...
inactive_level = 4.0
active_level = np.mean(full_df.POTENCY.values)
rf_beta = partial(np.interp, xp=bins, fp=result_hdr['rfb'])

# first fit full models....
reload(stats)
df_tcams_nd = pd.read_pickle('parsed/tcams_nodups.pkl')
full_models = stats.fit_models(df_tcams_nd, full_df)
from malaria import mp_utils

full_models['rfb'] = rf_beta
full_models['frac_act'] = frac_active
full_models['inactive_level'] = inactive_level
full_models['active_level'] = active_level


# In[90]:


# Adding minimum distance from training set data to the Molport data...
from malaria import mp_utils

#min_dist = np.ones(mp1.shape[0])
#mp1 = pd.read_pickle(std_files[0])
#mp1_fps = np.asarray([x for x in mp1.fps.values])
blk = []
smile_blk = []

def make_blocks():
    for i in range(1, full_df.shape[0], 1000):
        e = min(i+1000, full_df.shape)
        blk.append(np.asarray([x for x in full_df.fps.values[i:e]]))
        smile_blk.append([x for x in full_df.index[i:e]])
        pass

#make_blocks()

def add_min_dist(fn):
    mp_df = pd.read_pickle(fn)
    
    mins = np.ones(mp_df.shape[0], dtype=float)
    amins = np.ones(mp_df.shape[0], dtype=float)
    min_smiles = mp_df.SMILESIS.values.copy()
    mp1_fps = np.asarray([x for x in mp_df.fps.values])
    
    for (i, bl) in enumerate(blk):
        bl_min = np.min(cdist(bl, mp1_fps, metric='jaccard'), axis=0)
        amin = np.argmin(cdist(bl, mp1_fps, metric='jaccard'), axis=0)
        
        min_smiles[bl_min < mins] = np.asarray(smile_blk[i])[amin[bl_min < mins]]
        mins = np.minimum(mins, bl_min)
    
    md = pd.Series(mins, index=mp_df.index)
    sm = pd.Series(min_smiles, index=mp_df.index)
    mp_df['min_dist'] = md
    mp_df['min_smile'] = sm
    mp_df.to_pickle(fn)
    
#for fn in std_files:
    #add_min_dist(fn)    


# Now choose the top compounds from Molport accoriding to various criteria...
# 
# ### Figure 5.

# In[91]:


Artemisinin = 'CC1CCC2C(C(=O)OC3C24C1CCC(O3)(OO4)C)C'
Atovaquone  = 'C1CC(CCC1C2=CC=C(C=C2)Cl)C3=C(C4=CC=CC=C4C(=O)C3=O)O'
Chloroquine = 'CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl'
Doxycycline = 'CC1C2C(C3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O'
Mefloquine  = 'C1CCNC(C1)C(C2=CC(=NC3=C2C=CC=C3C(F)(F)F)C(F)(F)F)O'
Primaquine  = 'CC(CCCN)NC1=C2C(=CC(=C1)OC)C=CC=N2'
Piperaquine = 'C1CN(CCN1CCCN2CCN(CC2)C3=C4C=CC(=CC4=NC=C3)Cl)C5=C6C=CC(=CC6=NC=C5)Cl'
drugs = [Artemisinin, Atovaquone, Chloroquine, Doxycycline, Mefloquine, Primaquine, Piperaquine]
drugs = [standardise.run(drug) for drug in drugs]
drug_labels = ['Artemisinin', 'Atovaquone', 'Chloroquine', 
               'Doxycycline', 'Mefloquine', 'Primaquine', 'Piperaquine']

# Figure pot_hist in the paper.
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 20))
plt.subplot(211)
fig = Chem.Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in drugs], 
                          legends=drug_labels, molsPerRow=4)
plt.imshow(fig)
plt.xticks([])
plt.yticks([])
plt.title('Standard Drugs')
plt.subplot(212)
fig = mp_utils.summarize('POT', 5)
plt.imshow(fig)
plt.xticks([])
plt.yticks([])
_ = plt.title('Most potent compounds in our dataset')
savefig('figures/fig5_drugs_n_pot.png')


# ### Figure 6.

# In[92]:


# Figure pot_hist in the paper.
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 20), dpi=400)
plt.subplot(211)
fig = mp_utils.summarize('BEST_Q', 5)
plt.imshow(fig)
plt.xticks([])
plt.yticks([])
plt.title('Best potency, with bounds on Toxicity, Lipophilicity and QED')
plt.subplot(212)
fig = mp_utils.summarize('BEST_D', 5)
plt.imshow(fig)
plt.xticks([])
plt.yticks([])
_ = plt.title('... and different from existing drugs...')
savefig('figures/fig6.png')


# ### Figure 7.

# In[93]:


# Figure pot_hist in the paper.
def get_neighbour_labels(vals, min_dists):
    labels = []
    for (i, v) in enumerate(vals):
        it = full_df.loc[v]
        lbl = 'P:%.2f L:%.2f Q:%.2f %s' % (it.POTENCY, it.crippen, it.qed, min_dists[i])
        labels.append(lbl)
        pass
    return labels

plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 20), dpi=400)
plt.subplot(211)
fig = mp_utils.summarize('BEST_D', 5, check_new=full_df, min_dist=0.2)
plt.imshow(fig)
plt.xticks([])
plt.yticks([])
plt.title(".. and different from any we've seen in training")
neighbours, mds = mp_utils.summarize('BEST_D', 5, check_new=full_df, neighbours=True, min_dist=0.2, draw=False)
nl = get_neighbour_labels(neighbours, mds)
plt.subplot(212)
fig = Chem.Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in neighbours],legends=nl)
plt.imshow(fig)
plt.xticks([])
plt.yticks([])
_ = plt.title('... together with their closest counterparts in training set.')
savefig('figures/fig6.png')


# In[94]:


pwd


# ### Bias 2.

# In[95]:


tc_dists = cdist(fpa, fpa, metric='jaccard') + np.diag(np.ones(len(fpa)) * np.nan)
import glob
std_files = glob.glob('/Users/oliverwatson/evartech/molport/standardn_*')
df0 = pd.read_pickle(std_files[0])
idcs = np.random.choice(np.arange(df0.shape[0]), 10000)
fpr = np.asarray([x for x in df0.iloc[idcs].fps.values])
r_dists = cdist(fpr, fpr, metric='jaccard') + np.diag(np.ones(len(fpr)) * np.nan)
tcr_dists = cdist(fpr, fpa, metric='jaccard')
mal_hist= np.histogram(tc_dists[~np.isnan(tc_dists)], density=True, bins=7)
mr_hist = np.histogram(tcr_dists[~np.isnan(tcr_dists)], density=True, bins=7)

# magic number - but justified by looking at the bottom plot below.


# In[ ]:


mtc_dists = np.nanmin(tc_dists, axis=1)


# In[ ]:


mrtcr_dists = np.nanmin(cdist(fpr, fpa, metric='jaccard'), axis=1)


# In[ ]:


_ = plt.hist(mtc_dists, density=True, bins=20)
plt.grid(True)
_ = plt.title('Minimum distances between a point in the Malaria dataset and other points in the dataset')


# In[ ]:


_ = plt.hist(mrtcr_dists, density=True, bins=20)
plt.grid(True)
_ = plt.title('Minimum distances between a random compound and active compounds')


# In[160]:


h1 = np.histogram(mtc_dists, bins=10)


# In[161]:


h2 = np.histogram(mrtcr_dists, bins=10)


# In[165]:


plt.plot((h1[1][1:] + h1[1][:-1])*.5 , (h1[0] / (2*num_inactive_per_active * h2[0] + 1.)), 
         label='Using distribution of minimum distances')
plt.plot(x_axis, mal_hist[0]/(num_inactive_per_active*mr_hist[0]), label='Using distribution of distances')
plt.grid(True)
plt.xlabel('Tanimoto Distance')
plt.ylabel('Fraction of actives given 1% population active')
plt.legend(loc='best')
_ = plt.title('Estimate of fraction of actives as function of distance from an active')


# In[ ]:




