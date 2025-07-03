import pickle
import os

ms_data = set(os.listdir('/workspace/MassSpec'))
id_data = set(os.listdir('/workspace/IsoDist'))
hsqc_data = set(os.listdir('/workspace/HSQC_NMR'))
c_data = set(os.listdir('/workspace/C_NMR'))
h_data = set(os.listdir('/workspace/H_NMR'))

index = pickle.load(open('/workspace/index.pkl', 'rb'))

for idx in index:
    fn = f'{idx}.pt'
    index[idx]['has_mass_spec'] = fn in ms_data
    index[idx]['has_hsqc'] = fn in hsqc_data
    index[idx]['has_iso_dist'] = fn in id_data
    index[idx]['has_h_nmr'] = fn in h_data
    index[idx]['has_c_nmr'] = fn in c_data

pickle.dump(index, open('/workspace/index.pkl', 'wb'))