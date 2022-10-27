#!/bin/bash
export SCRATCH_DIR=/scratch/$PBS_JOBID
export CONDA_DIR=$SCRATCH_DIR/miniconda32
export MLRECOCALO_REPO_DIR=$HOME/ml-reco-calo-sync

sendmail "mmazurekgda@gmail.com" <<EOF
subject:CIS CLUSTER: INITIALIZED $PBS_JOBID on $HOSTNAME
EOF

if [ -d "${CONDA_DIR}" ] 
then
    echo "-> Directory ${CONDA_DIR} already exists!"
    source $CONDA_DIR/bin/activate
    conda activate calo_nn
else
    echo "-> Getting conda 4.7.10..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh -O $SCRATCH_DIR/miniconda3.sh
    echo "-> Installing conda environemnt..."
    bash $SCRATCH_DIR/miniconda3.sh -b -p $CONDA_DIR || true
    source $CONDA_DIR/bin/activate
    echo "-> Installing conda with tensorflow..."
    if [[ $HOSTNAME == *"gpu"* ]]
    then
        echo "--> using options for GPUs"
        conda create -y --name calo_nn --file $MLRECOCALO_REPO_DIR/requirements_gpu_ncbj.txt
    else
        echo "--> using options for CPUs"
        conda create -y --name calo_nn --file $MLRECOCALO_REPO_DIR/requirements_cpu_ncbj.txt
    fi
    conda activate calo_nn
    pip install pyfiglet GitPython
    #cp -r ~/miniconda3 /scratch/miniconda3
    #echo "~/miniconda3 copied to /scratch/miniconda3"
    echo "-> Done!"
fi

export OUTPUT_AREA=$MLRECOCALO_REPO_DIR/evaluations/$(python -c "from datetime import datetime; print(datetime.utcnow().strftime(\"%Y%m%d_%H%M%S%f\"))")
mkdir -p $OUTPUT_AREA

# echo "Loading bashrc"
# 
#echo "Activating conda environment"
#conda activate calo_nn
#module load gcc/9.3.0


# REPO_DIR=${HOME}/ml-reco-calo-sync
# cd ${REPO_DIR}

#CONFIG=$(cat config.py)
#BACKBONE=$(cat calo_yolo/backbone.py)
#CALOYOLO=$(cat calo_yolo/calo_yolo.py)

MODEL_NAME=custom_32x32
CONFIG_FILE=datasets/toy/GaussinoCaloShowers__phase1_inner__18x18s36x36__64x64_1GeV__gamma__base/config.json
export CONFIG_FILE_PATH=$MLRECOCALO_REPO_DIR/$CONFIG_FILE

sendmail "michal.mazurek@cern.ch" <<EOF
subject:CIS CLUSTER: STARTED $PBS_JOBID on $HOSTNAME

REPO $MLRECOCALO_REPO_DIR
HOST $HOSTNAME
OUTPUTAREA $OUTPUT_AREA
Job params:

$(python -c "import os; import json; config_file =  open(os.getenv(\"CONFIG_FILE_PATH\")); data = json.load(config_file); print('\n'.join([ ':'.join([prop, str(value)]) for prop, value in json.loads(data).items()]))")

EOF

cd $MLRECOCALO_REPO_DIR
python run.py \
--training \
--mirrored_strategy \
--output_area=$OUTPUT_AREA \
--model_name=$MODEL_NAME \
--verbosity="DEBUG" \
--config_file=$CONFIG_FILE \
--epochs 1 \
2>&1 | tee ${OUTPUT_AREA}/stdout


sendmail "mmazurekgda@gmail.com" <<EOF
subject:CIS CLUSTER: FINISHED $PBS_JOBID on $HOSTNAME
EOF

