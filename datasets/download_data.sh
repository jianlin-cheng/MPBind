#!/bin/sh

#### PDB sequences
# wget https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz -O data/pdb_seqres.txt.gz
# extract pdb_seqres
# gunzip -f data/pdb_seqres.txt.gz > data/pdb_seqres.txt

# The sequence in fasta format for all entries can be downloaded at: https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz



# parameters for PDB data
MIRRORDIR=data/all_biounits
LOGFILE=pdb_logs
SERVER=rsync.rcsb.org::ftp_data
PORT=33444
FTPPATH=/biounit/PDB/divided/


# download
rsync -rlpt -v -z --delete --port=$PORT ${SERVER}${FTPPATH} $MIRRORDIR > $LOGFILE
