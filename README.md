# The Transformer MeshCNN - Team 3

We did some big changes - the most significant one is the self attention addition (with patching to overcome the memory issues).

* Test Accuracy on CUBES (average of best 5 in 100 epochs):

* Test Accuracy on HUMANS (average of best 5 in 100 epochs):


with almost same amount of parameters!
We get faster converges and higher stability then the original MeshCNN on CUBES.

## Our changes
 * self attention:
    * implemented based on patched self attention to save memory. 
    * The adaption of the self attention technique is straight forward as shown (in /models/layers/mesh_self_attention.py).
 
 * replaced neighborhood 2d convolution by fully connected layer which increased performence and network expesiveness without
 * replaced neighborhood symetric transformations (|a-c|, a+c, |b-d|, b+d, e) with simple average (as in GCN): (a+b+c+d+e)/5
 * 
 * changed the pooling critiria to first feature value only but it didn't helped - commented





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
