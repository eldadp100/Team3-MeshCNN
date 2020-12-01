# The Transformer MeshCNN - Team 3

We did some big changes - the most significant one is the self attention addition (with patching to overcome the memory issues).

We used the classification as benchmark and got improvement also on the segmentation task. 

* Test Accuracy on CUBES (average of best 5 in 100 epochs):

* Test Accuracy on HUMANS (average of best 5 in 100 epochs):


with almost same amount of parameters!
We get faster converges and higher stability then the original MeshCNN on CUBES.

## Our changes
We porposed the use of the following two layers that we adapt to the mesh scenario. The first one is self attention layer that has the problem of memory consuming - to handle that we used patched self attention. We implemented a multi head self attention. The second is to use LSTM to go over the mesh edges in some order and rout information from one edge to another. The idea of using LSTM to globalize the patched (local) self attention is first introduced here. We use a circular LSTM which is applying LSTM several times while we keep the state of the previous iteration and use it as a start to the next (different from bidirectional LSTMs but with same motivation).

 * self attention (/models/layers/mesh_self_attention.py):
    * implemented based on patched self attention to save memory. 
    * multi head.
 * LSTM
    * to globalize the self attention information routing
    * circular LSTM
 * replaced 2d convolution on neighborhood by fully-connected layer which increased performence and network expesiveness.
 * replaced neighborhood symetric transformations (|a-c|, a+c, |b-d|, b+d, e) with simple average (as in GCN): (a+b+c+d+e)/5
 * we used BN instead GN as BN known to outperform GN - as we don't separate among multiple GPUs (so GN is not needed in our case)
    * fixed bug in the original code related to BN
 * changed the pooling critiria to first feature value only but it didn't helped - commented




## Setup
look at the original repository for more info: ranahanocka/MeshCNN
### Installation
- Clone this repo:
```bash
git clone https://github.com/ranahanocka/MeshCNN.git
cd MeshCNN
```
- Install dependencies: [PyTorch](https://pytorch.org/) version 1.2. <i> Optional </i>: [tensorboardX](https://github.com/lanpa/tensorboardX) for training plots.
  - Via new conda environment `conda env create -f environment.yml` (creates an environment called meshcnn)
  
### 3D Classification on CUBES
Download the dataset
```bash
bash ./scripts/cubes/get_data.sh
```

Run training (if using conda env first activate env e.g. ```source activate meshcnn```)
```bash
bash ./scripts/cubes/train.sh
```

To view the training loss plots, in another terminal run ```tensorboard --logdir runs``` and click [http://localhost:6006](http://localhost:6006).

Run test and export the intermediate pooled meshes:
```bash
bash ./scripts/cubes/test.sh
```

Visualize the network-learned edge collapses:
```bash
bash ./scripts/cubes/view.sh
```

### 3D Shape Segmentation on Humans
The same as above, to download the dataset / run train / get pretrained / run test / view
```bash
bash ./scripts/human_seg/get_data.sh
bash ./scripts/human_seg/train.sh
bash ./scripts/human_seg/get_pretrained.sh
bash ./scripts/human_seg/test.sh
bash ./scripts/human_seg/view.sh
```
