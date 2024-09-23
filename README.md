You can start training the network in train.ipynb notebook
- We can load the pre-trained weights for the backbone and the vertex detection network from here (https://github.com/zorzi-s/PolyWorldPretrainedNetwork) and freeze them for training to only train the Matching Network

### Main Parts
- Backbone CNN (Pre-trained)
- Vertex Detection (1x1 Conv) (Pre-trained)
- NMS (Selects top 256 points) (Not Trainable)
- Optimal Matching (Attentional GNN) (Our main challenge for training)

### Contributions
- Applied random permutations on groundtruth permutation matrix (line 31-32-98-105-106 in dataset.py)
- Applied Sinkhorn algorithm in the matching step (Thanks to https://github.com/henokyen/henokyen_polyworld)
- Dataloader

### Issues
- Everything is ok till we want to create the polygons from the points (Top 256 predictions) and the predicted adjacency matrix (the adjacenecy matrix is predicted correctly based on the ground truth), I guess the main problem is the way we reconstruct the polygons; we have the coordinates of the points and the adjacency matrix, but we don't know for example which vertex is the point (x,y); it's v1, v2 or vn? Probably we need to assign an ID to each point