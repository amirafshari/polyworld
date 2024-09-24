import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import math

def scores_to_permutations(scores):
    """
    Input a batched array of scores and returns the hungarian optimized 
    permutation matrices.
    """
    B, N, N = scores.shape

    scores = scores.detach().cpu().numpy()
    perm = np.zeros_like(scores)
    for b in range(B):
        r, c = linear_sum_assignment(-scores[b])
        perm[b,r,c] = 1
    return torch.tensor(perm)




def permutations_to_polygons(perm, graph, out='torch'):
    B, N, N = perm.shape

    def bubble_merge(poly):
        s = 0
        P = len(poly)
        while s < P:
            head = poly[s][-1]

            t = s+1
            while t < P:
                tail = poly[t][0]
                if head == tail:
                    poly[s] = poly[s] + poly[t][1:]
                    del poly[t]
                    poly = bubble_merge(poly)
                    P = len(poly)
                t += 1
            s += 1
        return poly

    diag = torch.logical_not(perm[:,range(N),range(N)])
    batch = []
    for b in range(B):
        b_perm = perm[b]
        b_graph = graph[b]
        b_diag = diag[b]
        
        idx = torch.arange(N)[b_diag]

        if idx.shape[0] > 0:
            # If there are vertices in the batch

            b_perm = b_perm[idx,:]
            b_graph = b_graph[idx,:]
            b_perm = b_perm[:,idx]

            first = torch.arange(idx.shape[0]).unsqueeze(1)
            second = torch.argmax(b_perm, dim=1).unsqueeze(1).cpu()

            polygons_idx = torch.cat((first, second), dim=1).tolist()
            polygons_idx = bubble_merge(polygons_idx)

            batch_poly = []
            for p_idx in polygons_idx:
                if out == 'torch':
                    batch_poly.append(b_graph[p_idx,:])
                elif out == 'numpy':
                    batch_poly.append(b_graph[p_idx,:].numpy())
                elif out == 'list':
                    g = b_graph[p_idx,:] * 300 / 320
                    g[:,0] = -g[:,0]
                    g = torch.fliplr(g)
                    batch_poly.append(g.tolist())
                elif out == 'coco':
                    g = b_graph[p_idx,:] * 300 / 320
                    g = torch.fliplr(g)
                    batch_poly.append(g.view(-1).tolist())
                else:
                    print("Indicate a valid output polygon format")
                    exit()
        
            batch.append(batch_poly)

        else:
            # If the batch has no vertices
            batch.append([])

    return batch








def graph_to_vertex_mask(points, image):
    B, _, H, W = image.shape

    mask = torch.zeros((B, H, W), dtype=torch.uint8)

    # Loop style
    # for batch in range(B):
    #     mask[batch, points[batch, :, 0], points[batch, :, 1]] = 1

    # Vectorized Style
    batch_indices = np.arange(B)[:, None]
    mask[batch_indices, points[:, :, 0], points[:, :, 1]] = 1
    
    return mask





def polygon_to_vertex_mask(polygons: list):
    B = len(polygons)
    mask = torch.zeros((B, 320, 320), dtype=torch.uint8)

    for batch in range(B):
        batch_polygons = [np.array(poly, dtype=int) for poly in polygons[batch]]
        for poly in batch_polygons:
            for point in poly:
                mask[batch, point[1], point[0]] = 1


    return mask





def tensor_to_numpy(input: list):
    ''' convert a list of tensors to a list of numpy arrays '''
    numpy = []
    for batch in range(len(input)):
        batch_polygons = [tensor.cpu().numpy() for tensor in input[batch]]
        numpy.append(batch_polygons)
    return numpy





def point_to_polygon(points: list):
    
    B = len(points)
    polygons = []

    for batch in range(B):
        batch_polygons = [np.array(poly, dtype=int).reshape(-1, 2) for poly in points[batch]]
        polygons.append(batch_polygons)

    return polygons









def polygon_to_seg_mask(polygons, image_size):
    B = len(polygons)
    mask = np.zeros((B, image_size, image_size), dtype=np.uint8)
    for batch in range(B):
        for polygon in polygons[batch]:
        # for polygon in polygons[0]:
            cv2.fillPoly(mask[batch], [polygon.detach().cpu().numpy()], 1)
    return torch.tensor(mask, dtype=torch.float32, device=device, requires_grad=True)














def sort_sync_nsm_points(nms_graph, vertices, gt_index):
    ''' 
    nms graph: (B, N, 2)
    Vertices: numpy array of points (B, N, 2) --> N: number of unique points
    GT_Index: A Dictinary with points as its keys and their index as values (It's the same as vertices but with index values) (B, N)
    '''
    B, N, D = nms_graph.shape # 1, 256, 2
    sorted_nsm_points = np.zeros((B, N, D), dtype=int)
    nms_graph = nms_graph.detach().cpu().numpy()
    
    for b in range(B):
        sorted_nsm = np.zeros((N, D), dtype=int)

        n = vertices[b].shape[0] # Number of Viertices (Points)
        m = nms_graph[b].shape[0] # Number of Predicted Vertices (Points)
        # print(vertices[b].shape, nms_graph[b].shape)
        distances = np.linalg.norm(vertices[b][:, None] - nms_graph[b], axis=2) # Adds a new axis to vertices[b], changing its shape from (M, D) to (M, 1, D).
        distances = distances.reshape(n*m, 1)
        # print(distances.shape)
        distances = np.hstack((distances, np.repeat(vertices[b], m, axis=0),np.tile(nms_graph[b], (n, 1))))
        sorted_distance = distances[np.argsort(distances[:,0])]
        
        # Sort distances by the first column (distance)
        sorted_distances = distances[np.argsort(distances[:,0])]

        cndd_used = set()
        gt_used = set()
        cndd_mapped = {tuple(cndd):0 for cndd in nms_graph[b]}
       
        for d_p in sorted_distances:
            gt_p = tuple((d_p[1], d_p[2]))
            cndd_p = tuple((d_p[3], d_p[4]))
            if gt_p not in gt_used and cndd_p not in cndd_used:
                #print('we have a match ..', gt_p ,'->', cndd_p, ' with distance of ', d_p[0])
                # print(gt_p)
                sorted_nsm[gt_index[b][gt_p]]= list(cndd_p)
                gt_used.add(gt_p)
                cndd_used.add(cndd_p)
                cndd_mapped[cndd_p] = 1
                                
        restart_index = n
        for k, v in cndd_mapped.items():
            if v ==0:
                sorted_nsm[restart_index] = list(k)   
                restart_index +=1
        sorted_nsm_points[b] = sorted_nsm 
    return torch.from_numpy(sorted_nsm_points)





def prepare_gt_vertices(vertices, device='cuda', MAX_POINTS=256):
    B = len(vertices)
    v_gt = torch.empty((B, MAX_POINTS, 2), dtype=torch.float64)
    for b in range(B):
        gt_size = vertices[b].shape[0]
        extra = torch.full((MAX_POINTS - gt_size, 2), 0, dtype=torch.float64)
        extra_gt = torch.cat((vertices[b], extra), dim=0).to(device)
        v_gt[b] = extra_gt
    return v_gt.to(device)


    

def angle_between_points(A, B, C, batch=False):
    d = 2 if batch else 1
    AB = A - B
    BC = C - B
    epsilon = 3e-8
    
    AB_mag = torch.norm(AB, dim=d) + epsilon
    BC_mag = torch.norm(BC, dim=d) + epsilon
    
    dot_product = torch.sum(AB * BC, dim=d)
    cos_theta = dot_product / (AB_mag * BC_mag)
    
    zero_mask = (AB_mag == 0) | (BC_mag == 0)
    cos_theta[zero_mask] = 0  
    theta = torch.acos(torch.clamp(cos_theta, -1 + epsilon, 1 - epsilon))
    theta[zero_mask] = 0
    return theta * 180 / math.pi












def soft_winding_number(pred_polys, lam=1000, img_size=320, device='cuda'):

    B = len(pred_polys)
    IMG_SIZE = img_size
    pred_mask = torch.zeros((B, IMG_SIZE, IMG_SIZE)).to(device)
    
    x = torch.arange(IMG_SIZE)
    y = torch.arange(IMG_SIZE)
    xx, yy = torch.meshgrid(x,y)

    pixel_coords = torch.stack([yy, xx], dim=-1).float()
    
    for b in range(B):
        vertices  = torch.vstack(pred_polys[b]).float()
        #vertices = vertices.detach().cpu()
        #vertices.requires_grad=True
        #vertices =  vertices.unfold(dimension = 0,size = 2, step = 1)
        #vertices_repeated = vertices.repeat_interleave(IMG_SIZE*IMG_SIZE, dim=0)
        
        pairs = vertices[:-1].unsqueeze(1).repeat(1, 2, 1)
        pairs[:, 1, :] = vertices[1:]
        
        pairs_repeated = pairs.repeat_interleave(IMG_SIZE*IMG_SIZE, dim=0)
        
        #pixel_coords_angle = pixel_coords.repeat(vertices.shape[0],1,1).view(vertices.shape[0] *IMG_SIZE*IMG_SIZE,2)
        #pixel_coords_det = pixel_coords.repeat(vertices.shape[0],1,1).view(vertices.shape[0] *IMG_SIZE*IMG_SIZE ,2,1)
        pixel_coords_angle = pixel_coords.repeat(pairs.shape[0],1,1).view(pairs.shape[0] *IMG_SIZE*IMG_SIZE ,1 , 2).to(device)
        
        concatenated = torch.cat([pairs_repeated, pixel_coords_angle], dim=1)
        
        #ones = torch.ones(IMG_SIZE*IMG_SIZE*vertices.shape[0], 3).reshape(IMG_SIZE*IMG_SIZE*vertices.shape[0],1, 3)
        
        ones = torch.ones(pairs.shape[0] *IMG_SIZE*IMG_SIZE, 3, 1).to(device) #.reshape(vertices.shape[0]-1 *IMG_SIZE*IMG_SIZE,3, 1)
        output = torch.cat((concatenated, ones), dim=2)

        det = torch.det(output)

        # compute angle
        angles = angle_between_points(pairs_repeated[:, 0], pixel_coords_angle.view(pairs.shape[0] *IMG_SIZE*IMG_SIZE, 2), pairs_repeated[:, 1], batch=False)
       
        #Compute the soft winding number using equation 13
        w = (lam * det) / (1 + torch.abs(det *lam))
        w = w * angles

        w = w.view(pairs.shape[0], IMG_SIZE, IMG_SIZE)
        # Sum over all pairs of adjacent vertices to get the winding number
        w = w.sum(dim=0)
        
        pred_mask[b] = w

    return pred_mask















    


# def sinkhorn_knopp(cost_matrix, epsilon=0.05, iterations=100):
#     """
#     Sinkhorn-Knopp algorithm to approximate optimal transport
#     """
#     B, N, _ = cost_matrix.shape
#     log_mu = torch.zeros_like(cost_matrix)
#     log_nu = torch.zeros_like(cost_matrix)

#     u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
#     for _ in range(iterations):
#         u = epsilon * (torch.log(torch.ones(B, N, 1, device=cost_matrix.device)) - 
#                        torch.logsumexp((-cost_matrix + v) / epsilon, dim=-1, keepdim=True)) + log_mu
#         v = epsilon * (torch.log(torch.ones(B, 1, N, device=cost_matrix.device)) - 
#                        torch.logsumexp((-cost_matrix + u) / epsilon, dim=-2, keepdim=True)) + log_nu

#     return torch.exp((-cost_matrix + u + v) / epsilon)

# def scores_to_permutations(scores, temperature=0.1):
#     """
#     Input a batched array of scores and returns the approximate
#     permutation matrices using Sinkhorn-Knopp algorithm.
#     Preserves gradients for backpropagation.
#     """
#     B, N, _ = scores.shape
    
#     # Normalize scores to be non-negative
#     scores_normalized = scores - scores.min(dim=-1, keepdim=True)[0]
    
#     # Use Sinkhorn-Knopp to approximate permutation matrices
#     perm_soft = sinkhorn_knopp(-scores_normalized)
    
#     # Use a differentiable approximation of argmax
#     perm_hard = F.gumbel_softmax(torch.log(perm_soft + 1e-8), tau=temperature, hard=True, dim=-1)
    
#     return perm_hard