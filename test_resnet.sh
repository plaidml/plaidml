cd build-x86_64/Release 
#PLAIDML_VERBOSE=5 
#PYTHONPATH=$PWD python plaidbench/plaidbench.py -n3 keras resnet50
PLAIDML_DUMP=/nfs_home/stavarag/work/PlaidML/plaidml-brgemm_polydl/resnet50_files_hoisting PYTHONPATH=$PWD python plaidbench/plaidbench.py -n3 keras resnet50
cd ../..
