# Refrigeration

## Environment 1



## 🏗️ Structure

```bash
Refrigeration
├── env_config/ # YAML Environment Parameters
├── hyperparameters/ # YAML Hyperparameters for RL algorithm
├── scripts/ # Training and evaluation and policy scripts
├── src/ # Source code for environments, and Helpers
├── results/ # Trained agents , trajectories, evaluations
├── README.md # Project overview and instructions
└── requirements.txt # Python dependencies
 ```


## I. Installation

### Online RL 

1. Use conda or your favourite package manager.

    ```bash
    conda create -n online_rl python=3.11
    conda activate online_rl
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia #for linux and GPU! 
    conda install pytorch torchvision torchaudio -c pytorch #for mac 
    pip install -r requirements.txt
    ```
    NOTE:
    Make sure to save the path where *src/* is located every time you start a new terminal. Most scripts will contain code at the top like:

    ```python
    import os, sys
    ROOT = os.path.abspath(os.curdir)
    sys.path.append(os.path.abspath(os.path.join(ROOT,'src')))
    ```

    This handles the issue, but it can also be resolved directly as such:

    ```bash
    export PYTHONPATH="$PWD/src"
    ```

2. To run the RL algorithm choose the environment parameter file and the online RL algorithm with its hyperparameters. 
(The following code will choose C2_1 which is a first setup for environment 2 as the environment config and H1 has the algorithm name and hyperparameters.)
Before running this file make sure the train file has the same environment import as the one you want to try !!!
s is the starting random seed and l is the number of them (for now keep random seeds below 10! )


```bash
conda activate online_rl
python scripts/train.py --start_seed s --n_seeds l --HP_path hyperparameters/H1.yaml --C_path env_config/discrete_joint/C1.yaml 
```


3. To Run the policy and evaluation file, use the following code in terminal using your prefared values: [based on the model and policy name, change these values to run the oilicy on environment. Check the list of arguments in policy file! If you want to use the default value, you can skeep the argument. For list of numbers, write them with space!]

```bash
conda activate online_rl
python scripts/policy.py  --HP_path hyperparameters/H1.yaml --C_path env_config/discrete_joint/C1_10.yaml --E_path eval_config/E1.yaml 
```

## The Environments 1

The action space is on/off and T suction continuous. 

## III 

## IX. References 
#### gymnasium
- [Documentation](https://gymnasium.farama.org/)
- [Creating a custom env (FULL)](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py)
#### stablebaselines3
- [Documentation](https://stable-baselines3.readthedocs.io/en/master/)
#### wandb
- [Quickstart](https://docs.wandb.ai/quickstart/)

#### OFFLINE RL 

*** The requirement file is not supporting Offline RL yet ! (Some of these packages require numpy version that is not supported by stable-baseline3!)

1. [d3rlpy](https://github.com/takuseno/d3rlpy?tab=readme-ov-file#readme) : An offline deep RL library
2. [D4RL](https://github.com/Farama-Foundation/D4RL) : A collection of refrence environmets for offline RL
3. [Minari](https://github.com/Farama-Foundation/Minari) : A python library for conducting research in offline RL
4. [OfflineRL](https://github.com/polixir/OfflineRL.git) : A collection of offline RL algorithms
5. [corl](https://github.com/tinkoff-ai/CORL.git): High-quality single-file implementations of SOTA Offline and Offline-to-Online RL algorithms 

 