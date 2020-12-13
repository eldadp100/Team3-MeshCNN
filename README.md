# The Mesh Transformer - Team 3

Competition Results:
1. On CUBES dataset we got 98.9% test accuracy after 311 epochs (stable result). The average of top 5 in 100 epochs is 97% - the result is verfied in 2 separate executions.
2. On HUMANS dataset we got 92.4% test accuracy after 128 epochs and 92.6% after 313 epochs. The average of top 5 in first 100 epochs is 92.2% verfied on 2 separate executions.

We implemented 2 models - on is self attention adaptation to meshCNN which we call "Mesh Transformer" and one is LSTM based mesh walk.

The attention based model performed very well. I think we are the current SOTA on CUBES (Our test accuracy is 98.9% and the current SOTA is 98.6%. It's important to mention that our method is MeshCNN based and not related to the current SOTA - we got the results while going on different direction and therefore we do think it's an avidance that MeshCNN based model can still be the SOTA).

p.s. We didn't take any code from the web - we wrote all the code by ourselves.

Request: Because we inversted siginificant time in this we would like to continue working on this more and consider it as a part of the final project also (we inversted much more time than for just HW :) ) 

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
    
We use the notation of "Full transformer" and "Self Attention based pooling only" also in the table results. There we see that Self Attention based pooling only outperforms the full transformer.

Possible reasons:
1. We used patched self attention
2. We had to give it more training time
3. Not appropiate initialization
4. Using bigger datasets
5. Use matrix multiplication to aggreagate the heads as described in the paper. The problem is that it takes a lot of memory (we used small embedding size to overcome this issue but there it's not possible - there're options for that like 2 layers which will take less parameters but I saw that max operation also works well in practice so we preferd it. it might be the reason).

### The patching
Our mesh transformer supports patching to handle memory consumption.
Problems:
1. The self attention is very local - a lot of improvement can from globalizing the attention (as in the original form - window_size=num_of_edges).
Possible improvements:
1. Make the patches overlap.
2. Use better technique (e.g. low rank approximizations of exp(KQ^T)) to overcome the memory issue (e.g. Performers for low rank approx. LinFormer. LongFormer...).

### Hyperparameters
* Window size
* Embedding size
* Number of self attention heads

We added those to the command line options.

## LSTM based Mesh Walk
### General
At the time we tried that we didn't know that there's existing work that does it (which is also the current SOTA on CUBES) 

## Another changes we implemented
1. Adding edge embedding layer - We think it can help becasue we think that the initialized 5 features can disturb each other in the aggregation. When we tried it didn't increase the results but we still think is a good option   
2. Changing all the convolutions to fully connected layers (we thought it can increase the network expressiveness but turn out to perform bad)

## Small changes we tried and helped
1. Batch normalization instead Group normalization. We notice that we run on one GPU and Group norm is good when applying on multiple GPU (becasue then the batch is seperated on the GPUs and each has 1 or 2 samples - not accuracte mean and variance can be calculated). 
2. We run with larger batch size also (32 instead 16) which also helped (becasue of change 1)  

There was a tiny bug in the original code in the batch option ('BatchNorm2D' should be instead of 'BatchNorm' in norm selection).

## Small changes we tried on the original code and didn't helped 
1. Changing the aggregation in mesh convolution layer to average instead of symetric transformations and concatatnation. It decreased the results. 
2. Changing the pooling critiria to the norm of the first feature only - decreased results.
3. Adding dropout didn't helped - we used BN that known to replace it.


## Ciscular LSTM based Mesh Walk
LSTM to go over the mesh edges in some order and rout information from one edge to another. The idea of using LSTM was at start in order to globalize the patched (local) self attention (this approach doesn't exist and we thought is cool). Then we saw that with this layer only (and 0.6M parameters only) we got 80% accuracy on CUBES which proves there's something in this layer that can work. 

We use a circular LSTM which is applying LSTM several times while we keep the state of the previous iteration and use it as a start to the next (different from bidirectional LSTMs but with same motivation). The exact motivation here that we want the LSTM to first compute a global information and then use it as start hidden vector - because otherwise the first of elements of the sequence suffer from bad hidden state (that contains no information at start).

### Improvemnets Ideas
1. Make it to partial and many traverses (i.e. not on the whole mesh) - as done in Mesh Walk paper. 
2. As there is attention based Deep Walk we can do something similar here when we greedily traverse the graph and each time go the the nearby edge with highest attention degree (defined in MeshTransformer section). 
3. Use recent advances in Linear Multi Arm Bandits because this setting can be adapted to there.


## Results
### CUBES Results
Our best model which perform self attention based pooling (not a full transformer) is with test accuracy of 98.9% (and stable) - which we think is the current SOTA now.
We attach here a table with 
#### Results Discussion
* Our method passed the current SOTA by 0.3%. 
* We saw that full transformer decreased the results compared to using only attention based pooling. It emphesizes the importance of the pooling layer as when the self attention is dedicated to pooling only. 
* The good results apeared around epoch 200 where we also got stable results. The 98.9% apeared in epoch 311.
* We already discussed on our thoughts on how to improve the full transformer and the LSTM bsed mesh walk results.
There's more specific discuession in Notes column of the table.

#### Training Plots


### Human Segmentation Results
#### Results Discussion
We see that this benchmark is harder. We also had only 500 train samples which we think make the training process harder.
The highest score we got is 92.6% after 313 epochs. We don't know what is the current SOTA in this benchmark so we can't compare.
Here we see that higher window size does helped (in contrast to CUBES).
There's more specific discuession in Notes column of the table.
#### Training Plots

#### Pooling Visualizations
<p>
<img src="https://github.com/eldadp100/Team3-MeshCNN/blob/master/results_images/pool_0.jpeg" width="180">
<img src="https://github.com/eldadp100/Team3-MeshCNN/blob/master/results_images/pool_1.jpeg" width="150">
<img src="https://github.com/eldadp100/Team3-MeshCNN/blob/master/results_images/pool_2.jpeg" width="180">
<img src="https://github.com/eldadp100/Team3-MeshCNN/blob/master/results_images/pool_3.jpeg" width="165">
</p>

### Shrec Results
#### Training Plots

#### Discussion
We see that here the augmentation is important. More percise discussion in at Notes column.

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
