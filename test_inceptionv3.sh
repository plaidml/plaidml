MODEL=inception_v3
cd build-x86_64/Release 
#PLAIDML_VERBOSE=4 
PLAIDML_DUMP=/nfs_home/stavarag/work/PlaidML/plaidml-brgemm_polydl/${MODEL}_files PYTHONPATH=$PWD python plaidbench/plaidbench.py -n3 keras ${MODEL}
cd ../..
