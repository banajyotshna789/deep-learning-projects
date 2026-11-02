#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=caption
#SBATCH --output=output.txt

echo "Starting job..."
date

echo "Current directory:"
pwd

echo "Moving to project directory..."
cd <project-root>

echo "Now inside:"
pwd

echo "Listing files:"
ls

echo "Running training..."
python train.py

echo "Job finished"
date
