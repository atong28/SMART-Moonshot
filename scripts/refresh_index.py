import pickle
import os

#  TODO: Fix the hardcoded paths
DATA_ROOT = os.environ.get('DATA_DIR', '/data')

ms_data = set(os.listdir(f'{DATA_ROOT}/MassSpec'))
id_data = set(os.listdir(f'{DATA_ROOT}/IsoDist'))
hsqc_data = set(os.listdir(f'{DATA_ROOT}/HSQC_NMR'))
c_data = set(os.listdir(f'{DATA_ROOT}/C_NMR'))
h_data = set(os.listdir(f'{DATA_ROOT}/H_NMR'))

index = pickle.load(open(f'{DATA_ROOT}/index.pkl', 'rb'))

for idx in index:
    fn = f'{idx}.pt'
    index[idx]['has_mass_spec'] = fn in ms_data
    index[idx]['has_hsqc'] = fn in hsqc_data
    index[idx]['has_iso_dist'] = fn in id_data
    index[idx]['has_h_nmr'] = fn in h_data
    index[idx]['has_c_nmr'] = fn in c_data

pickle.dump(index, open(f'{DATA_ROOT}/index.pkl', 'wb'))