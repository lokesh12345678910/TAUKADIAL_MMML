#!/bin/bash
# -----------------------------------------------
# SLURM script to derive the final RACS acoustic feature set on a directory of audio files 
# ----------------------------------------------
#SBATCH -J trimmingTAUKADIAL       #name of JOb
#SBATCH -o trimmingTAUKADIAL.o%j   #name of std output file 
#SBATCH -e trimmingTAUKADIAL.e%jn  # name of stderr output file
#SBATCH -p development         
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-user=lokeshpugalenthi@utexas.edu
#SBATCH --mail-type=all
#SBATCH -A DBS23006 #DBS23006 for Stephanie, IRI22010 for Jessy
CUDA_VISIBLE_DEVICES=0 
conda activate racsPipeline
python ../../pipelinePythonScripts/trimAudioFile.py ../TAUKADIAL-24/train/ ./