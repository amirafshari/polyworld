import numpy as np
import random
from skimage import io
from skimage.transform import resize
import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import cv2

class CrowdAI(Dataset):


    def __init__(self, images_directory, annotations_path, window_size=320):

        self.IMAGES_DIRECTORY = images_directory
        self.ANNOTATIONS_PATH = annotations_path
        
        self.window_size = window_size
        self.max_points = 256

        # load annotation json
        with open(self.ANNOTATIONS_PATH) as f:
            self.annotations = json.load(f)
        
        self.images = pd.DataFrame(self.annotations['images'])
        self.labels = pd.DataFrame(self.annotations['annotations'])


        # Generate the Permutation
        # torch.manual_seed(0)
        # self.permutation = torch.randperm(self.max_points)




    
    def _shuffle_adjacency_matrix(self, A):
        """
        generates a new permutation for each sample (or batch) 
        and shuffles the adjacency matrix A accordingly
        """

        n = A.shape[0]
        torch.manual_seed(0)
        permutation = torch.randperm(n)

        shuffled_A = A[permutation, :][:, permutation]

        return shuffled_A, permutation
    


    '''
    def _shuffle_adjacency_matrix(self, A):
        """
        generates a new permutation for each sample (or batch) 
        and shuffles the adjacency matrix A accordingly
        """
        n = A.shape[0]
        shuffled_A = A[self.permutation, :][:, self.permutation]
        return shuffled_A, self.permutation[:n]
    '''


    def _shuffle_vector(self, v, permutation):
        return v[permutation]


    def _create_adjacency_matrix(self, segmentations, N=256):

        adjacency_matrix = torch.zeros((N, N), dtype=torch.uint8)
        dic = {}

        n, m = 0, 0
        graph = torch.zeros((N, 2), dtype=torch.float32) # to create the seg mask with permutation_to_polygon()
        vertices = torch.zeros((N, 2), dtype=torch.float32) # to sort the nms points in the training
        for i, polygon in enumerate(segmentations):
            for v, point in enumerate(polygon):
                if v != len(polygon) - 1:
                    adjacency_matrix[n, n+1] = 1
                else:
                    adjacency_matrix[n, n-v] = 1


                # We just use it to create the segmentation mask with permutation_to_polygon(), no other use
                graph[n] = torch.tensor(point)
                n += 1 # n must be incremented in this way due to the functionality of the permutation_to_polygon() function, so we use m for the dictionary

                if tuple(map(float, point)) not in dic:
                    vertices[m] = torch.tensor(point)
                    dic[tuple(map(float, point))] = m
                    m += 1



        # Permute the adjacency matrix
        # adjacency_matrix[:n,:n], permutation = self._shuffle_adjacency_matrix(adjacency_matrix[:n, :n])

        # Fill the diagonal with 1s
        for i in range(n, N):
            adjacency_matrix[i, i] = 1
        
        # Permute the graph
        # graph[:n] = self._shuffle_vector(graph, permutation)
        # graph = graph[:n]
        
        return adjacency_matrix, graph, vertices[:m], dic


        


    def _create_segmentation_mask(self, polygons, image_size):
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        for polygon in polygons:
            cv2.fillPoly(mask, [polygon], 1)
        return torch.tensor(mask, dtype=torch.uint8)



    def _create_vertex_mask(self, polygons, image_shape=(320, 320)):
        mask = torch.zeros(image_shape, dtype=torch.uint8)

        for poly in polygons:
            for p in poly:
                mask[p[1], p[0]] = 1

        return mask








    def __len__(self):
        return len(self.images)







    def __getitem__(self, idx):

        image = io.imread(self.IMAGES_DIRECTORY + self.images['file_name'][idx])
        image = resize(image, (self.window_size, self.window_size), anti_aliasing=True)
        image = torch.from_numpy(image)
        width, height = self.images['width'][idx], self.images['height'][idx]
        ratio = self.window_size / max(width, height)



        # Get the image ID
        image_id = self.images['id'][idx]
        # Get all annotations for this image
        image_annotations = self.labels[self.labels['image_id'] == image_id]
        # get all polygons for the image
        segmentations = image_annotations['segmentation'].values
        segmentations = [e[0] for e in segmentations]
        for i, poly in enumerate(segmentations):
            # rescale the polygon
            poly = [int(e * ratio) for e in poly]
            # out of bounds check
            for j, e in enumerate(poly):
                if j % 2 == 0:
                    poly[j] = min(max(0, e), self.window_size - 1)
                else:
                    poly[j] = min(max(0, e), self.window_size - 1)
            segmentations[i] = poly




        # print(segmentations)
        segmentations = [np.array(poly, dtype=int).reshape(-1, 2) for poly in segmentations] # convert a list of polygons to a list of numpy arrays of points
        # print(segmentations)




        # create permutation matrix
        # permutation_matrix, graph, permutation = self._create_adjacency_matrix(segmentations, N=self.max_points)
        # create the simple adjacency matrix
        permutation_matrix, graph, vertices, dic = self._create_adjacency_matrix(segmentations, N=self.max_points)
        # create vertex mask
        vertex_mask = self._create_vertex_mask(segmentations, image_shape=(self.window_size, self.window_size))
        # create segmentation mask
        seg_mask = self._create_segmentation_mask(segmentations, image_size=self.window_size)


        segmentations = [torch.from_numpy(poly) for poly in segmentations]





        return image, vertex_mask, seg_mask, permutation_matrix, segmentations, graph, vertices, dic