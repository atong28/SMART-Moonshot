#!/bin/bash

# Usage: bash examples/scripts/get_npmrd_smiles.sh
# Download the smiles csv files from the NPMRD website
OUTPUT_DIR="/data/nas-gpu/wang/atong/Datasets/Benchmark/smiles"
wget https://np-mrd.org/system/downloads/current/smiles_NP0000001_NP0050000.csv.gz -O ${OUTPUT_DIR}/smiles_NP0000001_NP0050000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0050001_NP0100000.csv.gz -O ${OUTPUT_DIR}/smiles_NP0050001_NP0100000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0100001_NP0150000.csv.gz -O ${OUTPUT_DIR}/smiles_NP0100001_NP0150000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0150001_NP0200000.csv.gz -O ${OUTPUT_DIR}/smiles_NP0150001_NP0200000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0200001_NP0250000.csv.gz -O ${OUTPUT_DIR}/smiles_NP0200001_NP0250000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0250001_NP0300000.csv.gz -O ${OUTPUT_DIR}/smiles_NP0250001_NP0300000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0300001_NP0350000.csv.gz -O ${OUTPUT_DIR}/smiles_NP0300001_NP0350000.csv.gz

# Unzip the files
for file in ${OUTPUT_DIR}/*.csv.gz; do
    gunzip $file
done

# Concatenate the files
cat ${OUTPUT_DIR}/*.csv > ${OUTPUT_DIR}/smiles.csv