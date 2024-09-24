import numpy as np
import torch.nn.functional as F
import torch
from utils.utils import angle_between_points, prepare_gt_vertices




def compute_l_angle_loss(gt_permutation_mask, 
                         vertices,
                         pred_permutation_mat, 
                         graph, device='cuda'):
    v_gt = prepare_gt_vertices(vertices, device=device)

    v_gt_1 = torch.matmul(gt_permutation_mask, v_gt)
    v_gt_2 = torch.matmul(gt_permutation_mask, v_gt_1)
    gt_angle = angle_between_points(v_gt, v_gt_1, v_gt_2)
    #torch.isnan(gt_angle).any()
    
    pred_permutation_mat = pred_permutation_mat.to(device)
    v_pred_1 = torch.matmul(pred_permutation_mat, graph)
    v_pred_2 = torch.matmul(pred_permutation_mat, v_pred_1)
    pred_angle = angle_between_points(graph, v_pred_1.float(), v_pred_2.float())
    #torch.isnan(pred_angle).any()
    
    return pred_angle, gt_angle



def cross_entropy_loss(sinkhorn_results, gt_permutation):
    ''' 
        It only considers the positive matches
        and tries to minimize the value of positive matches


        One considereation is the distribution of the vertices in the GT permutation matrix 
        to overcome the overfitting and help the model to learn the 
        correct permutation matrix


        One solution is:
        1- order the polygons, based on the number of vertices
        2- select first vertex of each polygon and assign it a unique ID
        3- repeat 2
        

        Another Solution:
        - Using a random permutation matrix/vector

     '''
    loss_match = -torch.mean(torch.masked_select(sinkhorn_results, gt_permutation == 1))
    return loss_match 





# def iou_loss_function(pred, gt):
#     B, H, W = gt.shape
#     iou = 0
#     for batch in range(B):
#         K = len(pred[batch]) # Number of polygons
#         batch_tensor = np.zeros((K, H, W), dtype=np.uint8)
#         for i, poly in enumerate(pred[batch]):
#             cv2.fillPoly(batch_tensor[i], [poly.detach().cpu().numpy()], 1)

#         batch_pred_mask = torch.sum(torch.tensor(batch_tensor), dim=0).permute(1,0)

#         # plt.imshow(batch_pred_mask)
#         # plt.show()
#         # plt.imshow(gt[batch])
#         # plt.show()

#         intersection = torch.min(batch_pred_mask, gt[batch])
#         union = torch.max(batch_pred_mask, gt[batch])
#         batch_iou = torch.sum(intersection) / torch.sum(union)
#         iou += batch_iou

#     return torch.tensor(1 - iou, requires_grad=True)



# def iou_loss_function(pred, gt):
#     B, H, W = gt.shape
#     total_iou = 0
    
#     for batch in range(B):
#         batch_pred = torch.zeros((H, W), device=gt.device)
#         for poly in pred[batch]:
#             # Convert polygon to mask
#             mask = torch.zeros((H, W), device=gt.device)
#             poly_tensor = poly.long()  # Ensure integer coordinates
#             mask[poly_tensor[:, 1], poly_tensor[:, 0]] = 1
#             mask = F.max_pool2d(mask.unsqueeze(0).float(), kernel_size=3, stride=1, padding=1).squeeze(0)
#             batch_pred = torch.max(batch_pred, mask)
        
#         plt.imshow(mask)
#         plt.show()
#         plt.imshow(gt[batch])
#         plt.show()
        
#         intersection = torch.sum(torch.min(batch_pred, gt[batch]))
#         union = torch.sum(torch.max(batch_pred, gt[batch]))
#         batch_iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
#         total_iou += batch_iou
    
#     avg_iou = total_iou / B
#     return 1 - avg_iou



# def iou_loss_function(pred_mask, gt_mask):
#     '''
#     pred_mask: (B, H, W)
#     gt_mask: (B, H, W)
#     '''
#     pred_mask = F.sigmoid(pred_mask)
#     intersection = torch.sum(torch.min(pred_mask, gt_mask))
#     union = torch.sum(torch.max(pred_mask, gt_mask))
#     iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
#     loss = 1 - iou
#     # loss.requires_grad = True
    
#     return loss # Return 1 - IoU to minimize



def iou_loss_function(pred_mask, target_maks):
    pred_mask = F.sigmoid(pred_mask)
    
    intersection = (pred_mask * target_maks).sum()
    union = ((pred_mask + target_maks) - (pred_mask * target_maks)).sum()
    iou = intersection / union
    iou_dual = pred_mask.size(0) - iou

    #iou_dual = iou_dual / pred_mask.size(0)
    iou_dual.requires_grad = True
    return torch.mean(iou_dual)