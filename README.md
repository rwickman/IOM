---
ArtifactType: nupkg, executable, azure-web-app, azure-cloud-service, etc. More requirements for artifact type standardization may come later.
Documentation: URL
Language: typescript, csharp, java, js, python, golang, powershell, markdown, etc. More requirements for language names standardization may come later.
Platform: windows, node, linux, ubuntu16, azure-function, etc. More requirements for platform standardization may come later.
Stackoverflow: URL
Tags: comma,separated,list,of,tags
---

# Real-time Omni-channel Order Fulfillment Intelligence

<!-- One Paragraph of project description goes here. Including links to other user docs or a project website is good here as well. This paragraph will be used as a blurb on CodeHub. Please make the first paragraph short and to the point.

You can expand on project description in subsequent paragraphs. It is a good practice to explain how this project is used and what other projects depend on it.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system. -->

### Installation

#### Generic Installation
Install the dependencies:
```code
pip install -r requirements.txt
```

If you want to use a GPU, you will need to install [CUDA Toolkit 11.1.0](https://developer.nvidia.com/cuda-11.1.0-download-archive).
To verify the CUDA installation was correct, you can run:
```code
python -c "import torch; torch.cuda.is_available()"
```


#### Linux (Ubuntu 18.04) Installation
This subsection will explain how to setup on a VM on Azure with Ubuntu 18.04 and Python 3.6.9.
(In my experiments I used Standard NC6_Promo (6 vcpus, 56 GiB memory).)


After you ssh into the VM, you will need to clone the [repo](https://dev.azure.com/dynamicscrm/OneCRM/_git/IOM.InventoryOptimization) and then cd into it.

For example if the directory is IOM.InventoryOptimization then:
```code
cd IOM.InventoryOptimization
```

Then, you will have to update and install Pip.
```code
sudo apt-get update
sudo apt-get install python3-pip
```

Next, install the dependencies:
```code
pip3 install -r requirements.txt
```

Finally, to make use of the GPU for PyTorch, you will need to install [CUDA Toolkit 11.1.0](https://developer.nvidia.com/cuda-11.1.0-download-archive):
```code
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
sudo sh cuda_11.1.0_455.23.05_linux.run
```
When running the second command, make sure to accept the ELUA and then select install.


## Experiments
### Simple 2 inventory node 2 SKU scenario
In this section I will go through the steps to train and evaluate the policies.

#### Description of arugments used in this demonstration
Here is a description of the commands used in this demonstation:
| Command     | Description |
| ----------- | ----------- |
| --ac_epochs | Number of epochs to train actor-critic model on collected experinces. |
| --episodes  | The number of episodes to simulate and train on. |
| --epsilon_decay   | Epsilon decay step used for decaying the epsilon value in epsilon-greedy exploration.|
| --eval   | Evaluate the policies rather than train. |
| --eval_episodes  | How many episodes will be used to evaluate the policies. |
| --eval_order_max | Max number of orders in an episode during evaluation. |
| --inv_loc   | Path to the JSON file containing inventory node locations.|
| --lr | Learning rate used for DRL models. |
| --max_inv_prod  | Max inventory for each product across all inventory nodes. |
| --min_exps | The minimum number of timesteps to run before training actor-critic over stored experience. |
| --num_skus  | Number of unique product SKUs.|
| --no_per  | Don't use Prioritized Experience Replay (PER) for DQN model. |
| --order_max | Max number of orders in an episode during training. |
| --plot | Plot training results. |
| --policy  | The policy to train.|
| --policy_dir  | The directory of the policies that will be used evaluation |
| --save_dir  | Directory to save the models.|
| --train_iter   | Number of training steps after each episode.|
| --vis   | Visualize the policies when evaluating them.|

A more complete list of can be found in code/main.py or by running `python code/main.py --help`.

<br>

Create a directory for the fulfillment algorithms (i.e., policies) in the root directory:
```code
mkdir policies_loc_1
```


<!-- In the data/ directory, contains a sample of cities and inventory node locations. Setting these parameters allow for the location of inventory nodes -->
#### Train the policies
I will train all the policies on 512 episodes of fulfilling orders. In each episode, demand nodes are generated until all the inventory nodes run out of inventory or a maxmium amount of orders has been fulfilled.


To train the primal-dual policy (implemented based on the paper *[Primal–Dual Algorithms for Order Fulfillment at Urban Outfitters, Inc.](https://pubsonline.informs.org/doi/10.1287/inte.2019.1013))
```code
python code/main.py --inv_loc data/loc_1.json --num_skus 2 --num_inv_nodes 2 --policy primal --episodes 512 --max_inv_prod 20 --save_dir policies_loc_1/primal_loc_1
```

<br>

To run the DQN approach that uses a simple FFN:
```code
python code/main.py --inv_loc data/loc_1.json --num_skus 2 --num_inv_nodes 2 --policy dqn --episodes 512 --train_iter 4 --max_inv_prod 20  --save_dir policies_loc_1/dqn_loc_1 --epsilon_decay 512 --order_max 32
```
<br>

To run the DQN Embedding based approach:
```code
python code/main.py --inv_loc data/loc_1.json --num_skus 2 --num_inv_nodes 2 --policy dqn_emb --episodes 512 --train_iter 4 --max_inv_prod 20  --save_dir policies_loc_1/dqn_emb_loc_1 --epsilon_decay 512 --order_max 32
```

To run the DQN Embedding based approach without using PER:
```code
python code/main.py --inv_loc data/loc_1.json --num_skus 2 --num_inv_nodes 2 --policy dqn_emb_no_per --episodes 512 --train_iter 4 --max_inv_prod 20  --save_dir policies_loc_1/dqn_emb_no_per_loc_1 --epsilon_decay 512 --order_max 32 --no_per
```

<br>

To run the Actor-critic based approach:
```code
python code/main.py --inv_loc data/loc_1.json --num_skus 2 --num_inv_nodes 2 --policy ac --episodes 1024 --max_inv_prod 20  --plot --save_dir policies_loc_1/ac_loc_1  --order_max 32 --ac_epochs 3 --min_exps 128
```

#### Evaluate the policies

To evaluate the trained policies:
```code
python code/main.py --inv_loc data/loc_1.json --num_skus 2 --num_inv_nodes 2 --eval --eval_episodes 64 --max_eval --policy_dir policies_loc_1
```

To visualize the trained policies:
```code
python code/main.py --inv_loc data/loc_1.json --num_skus 2 --num_inv_nodes 2 --eval --eval_episodes 1 --policy_dir policies_loc_1 --vis
```



### 5 Inventory Node 8 SKU DQNEmb
Create the directory for the policies:
```code
mkdir policies_loc_4
```

Create the primal-dual policy for use of imitation learining:
```code
 py code/main.py --inv_loc data/loc_4.json --city_loc data/cities_3.json --num_inv_nodes 5 --num_skus 8 --policy primal --episodes 512 --max_inv_prod 20 --save_dir policies_loc_4/primal_loc_4 --rand_max_prod --rand_inv_sku_lam
```
Create the DQNEmb policy:
```code
python code/main.py --inv_loc data/loc_4.json --city_loc data/cities_3.json --num_inv_nodes 5 --num_skus 8 --policy dqn_emb --episodes 512 --max_inv_prod 20 --train_iter 4 --plot --save_dir policies_loc_4/dqn_emb_loc_4 --rand_max_prod --rand_inv_sku_lam  --expert_dir policies_loc_4/primal_loc_4 --expert_pretrain 1024 --order_max 64 --epsilon_decay 256
```

Evaluate:
```code
py main.py --inv_loc data/loc_4.json --city_loc data/cities_3.json --num_skus 8 --num_inv_nodes 5 --eval --eval_episodes 32 --max_inv_prod 20 --policy_dir policies_loc_4 --rand_max_prod --rand_inv_sku_lam
```


### 50 Inventory Nodes 64 SKUs Abilation Study
As there are many parts, we did an abilation study to see what tha effect was removing certain components, while keeping the rest fixed.
```code
mkdir policies_loc_5
```

Create the primal-dual policy for use of imitation learining:
```code
 py code/main.py --inv_loc data/loc_5.json --city_loc data/cities_3.json --num_inv_nodes 50 --num_skus 64 --policy primal --episodes 512 --max_inv_prod 20 --save_dir policies_loc_5/primal_loc_5 --rand_max_prod --rand_inv_sku_lam --order_max 512
```

Baseline (using imitation learing only):
```code
py code/main.py --inv_loc data/loc_5.json --city_loc data\cities_3.json --num_inv_nodes 50 --num_skus 64 --policy dqn_emb --episodes 512 --max_inv_prod 20 --train_iter 4 --plot --save_dir policies_loc_5/dqn_emb --dqn_steps 10 --epsilon_decay 512 --order_max 32 --expert_dir policies_loc_5/primal_loc_5 --expert_pretrain 2048 --rand_max_prod --rand_inv_sku_lam
```

Not using randomization of the maximum product quantity and invetory SKU lambda value:
```code
py code/main.py --inv_loc data/loc_5.json --city_loc data\cities_3.json --num_inv_nodes 50 --num_skus 64 --policy dqn_emb_no_rand --episodes 512 --max_inv_prod 20 --train_iter 4 --plot --save_dir policies_loc_5/dqn_emb_loc_5_no_rand --dqn_steps 10 --epsilon_decay 512 --order_max 32 --expert_dir policies_loc_5/primal_loc_5 --expert_pretrain 2048
```

Not using imitation learning (i.e., pretrinaing on prior policy):
```code
py code/main.py --inv_loc data/loc_5.json --city_loc data/cities_3.json --num_inv_nodes 50 --num_skus 64 --policy dqn_emb_no_pretrain --episodes 512 --max_inv_prod 20 --train_iter 4 --plot --save_dir policies_loc_5/dqn_emb_loc_5_no_pretrain --dqn_steps 10 --rand_max_prod --rand_inv_sku_lam --epsilon_decay 512 --order_max 32
```

Not using multistep DQN:
```code
py code/main.py --inv_loc data/loc_5.json --city_loc data\cities_3.json --num_inv_nodes 50 --num_skus 64 --policy dqn_emb_no_multistep --episodes 512 --max_inv_prod 20 --train_iter 4 --plot --save_dir policies_loc_5/dqn_emb_loc_5_no_multistep --dqn_steps 1 --epsilon_decay 512 --order_max 32 --expert_dir policies_loc_5/primal_loc_5 --expert_pretrain 2048 --rand_max_prod --rand_inv_sku_lam
```

Not using margin loss for imitation learning:
```code
py code/main.py --inv_loc data/loc_5.json --city_loc data\cities_3.json --num_inv_nodes 50 --num_skus 64 --policy dqn_emb --episodes 512 --max_inv_prod 20 --train_iter 4 --plot --save_dir policies_loc_5/dqn_emb_no_margin --dqn_steps 10 --epsilon_decay 512 --order_max 32 --expert_dir policies_loc_5/primal_loc_5 --expert_pretrain 2048 --rand_max_prod --rand_inv_sku_lam --expert_lam 0
```

Evaluate the results (without showing random and naive baselines):
```code
py code/main.py --inv_loc data/loc_5.json --city_loc data/cities_3.json --num_skus 64 --num_inv_nodes 50 --eval --eval_episodes 16 --max_inv_prod 20 --policy_dir policies_loc_5 --rand_max_prod --rand_inv_sku_lam --no_rand_fulfill_eval  --no_naive_fulfill_eval --eval_order_max 512
```
To not show the primal-dual results, you will have to move them to a different directory. This will evaluate every policy over 16 episodes with a maximum of 512 orders each.


## Running the tests
To run the unittests, cd to the test directory and then run:
```code
python -m unittest <test.py>
```
where \<test.py\> should be replaced with one the test names.

For example:
```code
python -m unittest test_dqn_policy.py
```


<!-- 

### Installing

A step by step series of examples that tell you how to get a development environment running

1. Describe what needs to be done first

    ``` batch
    Give an example of performing step 1
    ```

2. And then repeat for each step

    ``` sh
    Another example, this time for step 2
    ```


### End-to-end tests

Explain what these tests test and why

```
Give an example
```

### Unit tests

Explain what these test and why

```
Give examples
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

Documenting some of the main tools used to build this project, manage dependencies, etc will help users get more information if they are trying to understand or having difficulties getting the project up and running.

* Link to some dependency manager
* Link to some framework or build tool
* Link to some compiler, linting tool, bundler, etc

## Contributing

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) which outlines all of our policies, procedures, and requirements for contributing to this project.

## Versioning and changelog

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](link-to-tags-or-other-release-location).

It is a good practice to keep `CHANGELOG.md` file in repository that can be updated as part of a pull request.

## Authors

List main authors of this project with a couple of words about their contribution.

Also insert a link to the `owners.txt` file if it exists as well as any other dashboard or other resources that lists all contributors to the project.

## License

This project is licensed under the < INSERT LICENSE NAME > - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc --> -->
