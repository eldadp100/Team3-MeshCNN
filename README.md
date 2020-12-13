# The Mesh Transformer - Team 3

Competition Results:
1. On CUBES dataset we got 98.9% test accuracy after 311 epochs (stable result). The average of top 5 in 100 epochs is 97% - the result is verfied in 2 separate executions.
2. On HUMANS dataset we got 92.4% test accuracy after 128 epochs and 92.6% after 311 epochs. The average of top 5 in first 100 epochs is 92.2% verfied on 2 separate executions.

We implemented 2 models - on is self attention adaptation to meshCNN which we call "Mesh Transformer" and one is LSTM based mesh walk.

p.s. We didn't take any code from the web - we wrote all the code by ourselves.

## Mesh Transformer
### General
We added self attention layer to meshCNN.
self attention is known to sufffer from high memory consuming problem - to handle that we used patched self attention. That means we devided the mesh edges into disjoint local sets and we applied self attention on each (same weights to all patches). The patch size is denoted by window size (and we put default unchangeable stride=window_size so the patches are disjoint). (The implementation of that is in /models/layers/mesh_self_attention.py).

### Self Attention based pooling
Lets define the "attention degree" of edge e as the sum of the attention scores that all edges gave to this edeg.
We use the attention degree as an optimized critiria to Edge Collapse (i.e. Pooling). (Implmented in /models/layers/mesh_pool_sa.py)

We see that using self attention only for pooling and not updating the input (i.e. after applying self attention we ignore the output and take only the attention matrix - as it's used to self attention pooling) performs way better then does changing the input (as in transformers in usual).

Just to clarify:
* suppose self attention layer is the function sa that takes as input the edges list denoted by x and returns new_x, attention_matrix.
* use it for pooling only:
    ```
      * _, attention_matrix = sa(x)
      * x = pooling_based_self_attention(x, attention_matrix)
    ```
* use it as "full transformer":
    ```
      * x, attention_matrix = sa(x)
      * x = pooling_based_self_attention(x, attention_matrix)
    ```
### Full transformer vs Self Attention based pooling only


### The patching
Our mesh transformer supports patching to handle memory consumption.
Problems:
1. The self attention is very local - a lot of improvement can from globalizing the attention (as in the original form - window_size=num_of_edges).
Possible improvements:
1. Make the patches overlap.
2. Use better technique (e.g. low rank approximizations of exp(KQ^T)) to overcome the memory issue (e.g. Performers for low rank approx. LinFormer. LongFormer...).

### Hyperparameters


## Our changes
We porposed the use of the following two layers that we adapt to the mesh scenario. The first one is self attention layer that has the problem of memory consuming - to handle that we used patched self attention. We implemented a multi head self attention. The second is to use LSTM to go over the mesh edges in some order and rout information from one edge to another. The idea of using LSTM to globalize the patched (local) self attention is first introduced here. We use a circular LSTM which is applying LSTM several times while we keep the state of the previous iteration and use it as a start to the next (different from bidirectional LSTMs but with same motivation).
We think that combining those ideas with the original mesh convolutional layer might outperform the current SOTA as they give another directions to adapt classical ideas to meshes.

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

Full transformer vs self attention on pooling only:
========= Explain here ==================

## Results on CUBES
The accuracy in the table is an average of the 5 best in 200 epochs (the results for the competion shown at start). 
The following table shows the results we got on several hyperparams and different models. The changes in the hyperparameters are colored to make it more readable. 
The notes column is for discussion on the results we got as we changed the parameters.



The metioned accuracy is the average of best 5 over 150 epochs executed twice (and average taken). We also mention the accuracy on 100 epochs.
### Original
The original code test accuracy is 93%.
### Bottom Line
Our best model which perform self attention based pooling (and that it - not a full transformer) with test accuracy of 98% (and stable) and the average of best 5 up to 150 is greater than 97%. On 100 epochs it's 96.1%
### LSTM
we see that by using the porposed LSTM layer only (without using anything else) we got test accuracy of 82%

## CUBES Results Discussion
Our method is near the current SOTA. We think that with more hyperparamter tuning we can pass the SOTA (this result is the first we get).
We saw that full transformer decreased the results compared to using only attention based pooling. It emphesizes the importance of the pooling layer as when the self attention is dedicated to pooling only. The good realy results apeared after epoch 100 and around epoch 200 we got stable results (greater than 98%) and we didn't executed the full transformer for so long. Therefore, we think that this can be the reason for that applying self attention only to the pooling layer overcomed the full transformer (which also changes the input).



## Results on Human Segmentation
<p>
<img src="https://github.com/eldadp100/Team3-MeshCNN/blob/master/results_images/pool_0.jpeg" width="180">
<img src="https://github.com/eldadp100/Team3-MeshCNN/blob/master/results_images/pool_1.jpeg" width="150">
<img src="https://github.com/eldadp100/Team3-MeshCNN/blob/master/results_images/pool_2.jpeg" width="180">
<img src="https://github.com/eldadp100/Team3-MeshCNN/blob/master/results_images/pool_3.jpeg" width="165">
</p>



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
