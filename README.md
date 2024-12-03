# Learning Phase Manifolds in Music
The project aims to adapt the DeepPhase architecture for motion
(Stark et al.) [1] to learn the phase manifolds of musical pieces
in an unsupervised manner. We aim to invoke latent channels
that capture the non-linear periodicity in a song, from which we
can reconstruct the input. 

Work in progress: see this [document](https://docs.google.com/document/d/19ITWZiDgPPprBpS5jXa8sgFdNbslxxW3evR3Zupn0YU/edit?tab=t.0)


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

## References

1. [Periodic Autoencoders for Learning Motion Phase Manifolds](https://dl.acm.org/doi/10.1145/3528223.3530178)