
from __future__ import division, print_function

import pandas as pd
import numpy as np

from rdkit.Chem import Crippen
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
import joblib
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from standardiser import standardise


def get_crippen(x):
    """
    Get the logP value for molecule X
    :param x: Molecule
    :return: float: logP
    """
    try:
        cp = Crippen.MolLogP(x)
    except:
        return np.nan
    return cp

def get_qed(mol):
    """
    Get QED value (or nan if no molecule)
    :param mol: molecule
    :return: float QED
    """
    if mol is None:
        return np.nan
    return qed(mol)

def prepare_isidro_chemb_data(dump_file='parsed/chembl_isidro.pkl'):
    """
    This reads the data that Isidro gets off his query
    :param dump_file:
    :return:
    """

    full_list = pd.read_pickle('parsed/mal_B.pickle')


    pd_dict = {'SMILESIS': [], 'CHEMBL_SMILE': [],  'VAL': [], 'fps' : [], 'hfps' : []}

    for item in full_list:

        if item['pchembl_value'] is None:
            continue

        cs = get_canonical_smile(item['canonical_smiles'])
        fp = get_fp(cs)
        hfp = get_fp(cs, hashed=True)
        pd_dict['SMILESIS'].append(cs)
        pd_dict['CHEMBL_SMILE'].append(item['canonical_smiles'])
        pd_dict['VAL'].append(float(item['pchembl_value']))
        pd_dict['fps'].append(fp)
        pd_dict['hfps'].append(hfp)
        # Already checked that in all teh
        pass

    df = pd.DataFrame.from_dict(pd_dict)
    df.to_pickle(dump_file)
    return


def prepare_cheml_malaria_data(dump_file='parsed/chembl_all_pkl'):
    """
    This reads the chembl data from raw file.
    :return: None
    """

    df = pd.read_csv('/Users/oliverwatson/evartech/moarf/raw_data/chemblntd_all.txt.gz', sep='\t')
    df['SMILESIS'] = pd.Series([get_canonical_smile(x) for x in df.SMILES.values])
    fps = [get_fp(x) for x in df.SMILESIS.values]
    hfps = [get_fp(x, hashed=True) for x in df.SMILESIS.values]
    df['fps'] = pd.Series([np.asarray(fp) for fp in fps], index=df.index)
    df['hfps'] = pd.Series([np.asarray(fp) for fp in hfps], index=df.index)
    df.to_pickle(dump_file)



def prepare_full_plasmodium_f_data_from_chembl(chembl_dump='parsed/chembl_webclient_plasfalc_tgt',
                                               parsed_dump='parsed/chembl_data_parsed.pkl'
                                               ):
    """
    This collects all the data in Chembl for target=Plasmodium Falciparum, an
    :param chembl_dump: Raw data out of chembl.
    :param parsed_dump: Parsed data with our canonical smiles, filtered for proper activity values
    (though without any constraint on the relation).
    :return: None
    """

    from chembl_webresource_client.new_client import new_client

    act = new_client.activity.filter(target_organism__in=['Plasmodium falciparum',])
    selected = [x for x in act]
    joblib.dump(selected, chembl_dump)

    pd_dict = {'SMILEIS': [], 'CHEMBL_SMILE': [], 'CHEMBL_ID': [], 'VAL': [], 'REL': []}

    for item in selected:
        if item['canonical_smiles'] is None:
            continue
        if item['pchembl_value'] is None:
            continue

        try:
            val = float(item['pchembl_value'])
        except IndexError:
            continue

        is_sml = get_canonical_smile(item['canonical_smiles'])
        if is_sml == 'None':
            continue

        pd_dict['SMILEIS'].append(is_sml)
        pd_dict['CHEMBL_ID'].append(item['molecule_chembl_id'])
        pd_dict['VAL'].append(val)
        pd_dict['REL'].append(item['relation'])
        pd_dict['CHEMBL_SMILE'].append(item['canonical_smiles'])
        pass

    df = pd.DataFrame.from_dict(pd_dict)
    df.to_pickle(parsed_dump)
    return



def get_canonical_smile(x):
    """
    Make our smiles canonical
    :param x: smile (string)
    :return: canonical smile (string)
    """
    try:
        return standardise.run(x)
    except Exception:
        return 'None'
    
def get_fp(sm, logfail=False, hashed=False):
    """

    :param sm:
    :param logfail:
    :param hashed: Return hashed fingerprints a uint16 instead of
    :return: boolean fingerprint or uint16 hashed fingerprint
    """
    try:
        if hashed:
            fp = np.array(list(GetHashedMorganFingerprint(Chem.MolFromSmiles(sm),2,nBits=128)), dtype=np.uint16)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sm),2,nBits=128) 
    except:
        if logfail:
            print("Cannot extract Mol from %s" % sm)
            pass
        if hashed:
            fp = np.zeros(128, dtype=np.uint16)
        else:
            fp = np.zeros(128, dtype=bool)
            pass
        pass
    return fp



