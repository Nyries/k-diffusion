# k-diffusion fork for an internship

Fork from k-diffusion model [](https://github.com/Nyries/k-diffusion/tree/master) with improvements such as cascaded sampling and [*Analyzing and Improving the Training Dynamics of Diffusion Models*](https://arxiv.org/abs/2312.02696)

## Usage

The main files to use from user perspective are `train.py`, `convert_for_inference.py` and `sample.py`. The commands to use these files are recorded in `Commands.txt`. 
I recommend to create folders **Demo**, **Checkpoint** and **Samples** as I customed the training and sampling files to save their results in folder named like that.

## Location of models

The model is located in `k-diffusion` folder and architecture in `k-diffusion/models`. The baseline architecture from k-diffusion model remains unchanged. However there are some classes and functions added in `layer.py`, `utils.py`, `config.py` etc...

The new architecture from ADM is located in `configA.py` and the improvements implemented from [*Analyzing and Improving the Training Dynamics of Diffusion Models*](https://arxiv.org/abs/2312.02696) are in the next files (`configB.py`,...)
