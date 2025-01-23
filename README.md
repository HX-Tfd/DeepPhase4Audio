# Learning Phase Manifolds in Music
The project aims to adapt the DeepPhase architecture for motion
(Stark et al.) [1] to learn the phase manifolds of musical pieces
in an unsupervised manner. We aim to invoke latent channels
that capture the non-linear periodicity in a song, from which we
can reconstruct the input. 


## Setup
### Prerequisites
Required Packages (see ```requirements.txt```):
- Python 3.12.3
- matplotlib==3.8.4
- numpy==2.1.3
- Plotting==0.0.7
- pytorch_lightning==2.4.0
- scikit_learn==1.4.2
- torch==2.5.1
- rich >= 10.2.2
  
Required modules on Slurm:
  - `gcc/11`
  - `cuda/12.1`

### Steps
- **Local development**: To use the repository locally, simply clone it and install the dependencies inside a virtual environment. 
- **Running on a Slurm cluster**: This code is also designed to be compatible with a Slurm cluster environment. Clone the repository in your cluster and run ```./setup.sh``` for automatically creating the environment and adding the modules.
- **How to run the code**: ./train_slurm.sh. Replace the config file inside train_slurm.sh with the config file for the model you want to run.

### Models and Configurations

This repository contains two models proposed in our paper: PAEFlat and VQ-PAE. The implementation for PAEFlat can be found in the paewave branch, while the implementation for VQ-PAE is located in the model/vq-pae branch. Each branch includes the respective model files and their corresponding configuration files, enabling easy replication and experimentation.


## References

1. [Periodic Autoencoders for Learning Motion Phase Manifolds](https://dl.acm.org/doi/10.1145/3528223.3530178)
2. [PyTorch implementation of WaveNet](https://github.com/vincentherrmann/pytorch-wavenet/tree/master)
