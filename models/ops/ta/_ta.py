import math
import torch 
import torch.nn.functional as F

def compute_importance_scores(x: torch.Tensor, mode: str = 'W'):
    # compute each vector max (along channel) reducing to B x H x W
    # compute the max along the rows and cols 
    # x: B x H x W x C
    
    if x.ndim < 4:
        x.unsqueeze_(0)
    
    # get the max along the channel dimension, C 
    feats_max = x.max(dim=-1)[0] # B, H, W

    if mode == 'H':
        return feats_max.max(dim=2, keepdim=True)[0] # B, H, 1
    
    elif mode == 'W':
        return feats_max.max(dim=1, keepdim=True)[0] # B, 1, W
    
def compute_distance(
        q: torch.Tensor, 
        kv: torch.Tensor,
        mode: str = "cosine") -> torch.Tensor:

    if mode == "cosine":
        q_n = F.normalize(q, dim=-1)
        kv_n = F.normalize(kv, dim=-1)
        sim = torch.matmul(q_n, kv_n.transpose(1, 2))  
        dist = 1 - sim                                 
    elif mode == "l2":
        dist = torch.cdist(q, kv, p=2)
    else:
        raise ValueError("mode must be 'cosine' or 'l2'")
    return dist


def group_tokens_by_imp_score(
        x: torch.Tensor,
        rh: float = 0.25,
        rw: float = 0.25,
        rp_h: float = 0.7,
        rp_w: float = 0.7,
        device='cpu'
        ):
    
    B, H, W, D = x.shape
    x = x.to(device)

    # compute the importance scores for each row and column
    # row_scores: B x H x 1; col_scores: B x 1 x W

    row_scores = compute_importance_scores(x, mode='H')
    col_scores = compute_importance_scores(x, mode='W')

    # sort the scores and get the indexes
    row_indexes = row_scores.argsort(dim=1, descending=True)
    col_indexes = col_scores.argsort(dim=2, descending=True)

    # determine the number of tokens to keep for each dimension
    # rh and rw is the ratio of rows and columns to drop
    # e.g if H = W = 14, rh = rw = 0.25, then kept_rows = kept_cols = 11
    kept_rows = H - math.floor( H * rh ) 
    kept_cols = W - math.floor( W *  rw ) 

    # H * rp_h -> 14 * 0.7 = 9.8 -> 9
    repr_rows = math.floor( kept_rows * rp_h ) # rp_h and rp_w is the ratio of representative tokens
    repr_cols = math.floor( kept_cols * rp_w )

    # unique_tokens 11 - 9 = 2
    uniq_rows = kept_rows - repr_rows
    uniq_cols = kept_cols - repr_cols
    
    # split the indexes into three groups: representative, unique, normal
    # unique_tokens: 9: 9 + 2. these are the immediate tokens, most important after the representative tokens
    uniq_row_indexes = row_indexes[:, repr_rows: repr_rows + uniq_rows, :]
    uniq_col_indexes = col_indexes[:, :, repr_cols: repr_cols + uniq_cols]

    # normal tokens: are the remaining tokens after the unique tokens 
    norm_row_indexes = row_indexes[:, repr_rows + uniq_rows:, :]
    norm_col_indexes = col_indexes[:, :, repr_cols + uniq_cols:]

    # representative tokens: top-k most important tokens
    repr_row_indexes = row_indexes[:, :repr_rows, :]
    repr_col_indexes = col_indexes[:, :, :repr_cols]

    # print(f'kept rows: {kept_rows}, kept cols: {kept_cols}')
    # print(f'repr rows: {repr_rows}, repr cols: {repr_cols}')
    # print(f'uniq rows: {uniq_rows}, uniq cols: {uniq_cols}')

    # print(f'Unique tokens (row x col): {(
    #     uniq_row_indexes.shape, uniq_col_indexes.shape
    #     )}')
    
    # print(f'Representative tokens (row x col): {(
    #     repr_row_indexes.shape, repr_col_indexes.shape
    #     )}')
    
    # print(f'Normal tokens (row x col): {(
    #     norm_row_indexes.shape, norm_col_indexes.shape
    #     )}')

    # create the token group labels
    # goal is to recreate the H x W grid with labels for each token
    x_labs = torch.zeros((B, H, W), dtype=x.dtype)
    # print(f'x_labs: {x_labs.shape}')
    src = torch.ones(size=(1, 1, 1))

    # 0: representative tokens 
    x_labs.scatter_(
        dim=1,
        index=repr_row_indexes.expand(-1, -1, W),
        src=(0 * src).expand(B, repr_row_indexes.shape[1], W)
    )

    x_labs.scatter_(
        dim=2,
        index=repr_col_indexes.expand(-1, H, -1),
        src=(0 * src).expand(B, H, repr_col_indexes.shape[2])
    )

    # 1: unique tokens 
    x_labs.scatter_(
        dim=1,
        index=uniq_row_indexes.expand(-1, -1, W),
        src=(1 * src).expand(B, uniq_row_indexes.shape[1], W)
    )

    x_labs.scatter_(
        dim=2,
        index=uniq_col_indexes.expand(-1, H, -1),
        src=(1 * src).expand(B, H, uniq_col_indexes.shape[2])
    )

    # 2: normal tokens 
    x_labs.scatter_(
        dim=1,
        index=norm_row_indexes.expand(-1, -1, W),
        src=(2 * src).expand(B, norm_row_indexes.shape[1], W)
    )

    x_labs.scatter_(
        dim=2,
        index=norm_col_indexes.expand(-1, H, -1),
        src=(2 * src).expand(B, H, norm_col_indexes.shape[2])
    )
    
    # example data of x_labs[0] at batch 0
    #  [
    #      [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #      [1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 1., 1., 2., 1.],
    #      [1., 0., 0., 0., 1., 0., 0., 0., 2., 2., 0., 0., 2., 0.],
    #      [1., 0., 0., 0., 1., 0., 0., 0., 2., 2., 0., 0., 2., 0.]
    #   ]
    # print(x_labs)
    return x_labs

def inject_q_kv(q: torch.Tensor, kv: torch.Tensor, return_idx=False):

    q_norm = F.normalize(q, dim=-1)
    kv_norm = F.normalize(kv, dim=-1)

    # the simialritty matrix between the representative (q) tokens and normal (kv)
    sim = torch.matmul(q_norm, kv_norm.transpose(1, 2))

    # pick best kv index for each q: (B, Nq)
    idx = sim.argmax(dim=-1)

    # gather the values from kv matched with the representative (queries)
    values = torch.gather(kv, 1, idx.unsqueeze(-1).expand(-1, -1, kv.size(-1)))

    counts = torch.zeros(q.size(0), kv.size(1), device=q.device, dtype=torch.long)
    for b in range(q.size(0)):
        counts[b].scatter_add_(0, idx[b], torch.ones_like(idx[b]))
    
    if return_idx:
        return values, counts, idx

    return values, counts


def ejector_q_kv(
        x_reduced: torch.Tensor, 
        dist: torch.Tensor, 
        threshold_value: float = 0.5):
    
    B, N, M = dist.shape
    device = dist.device
    dtype = x_reduced.dtype

    mask = dist <= threshold_value                            
    empty = ~mask.any(dim=-1, keepdim=True)                   
    nearest = dist.argmin(dim=-1, keepdim=True)               
    fallback = torch.zeros(B, N, M, device=device, dtype=torch.bool)
    fallback.scatter_(dim=-1, index=nearest, value=True)      
    mask = torch.where(empty, fallback, mask)                

    weights = mask.to(dtype)                                  
    x_restore = torch.matmul(weights, x_reduced)                    
    counts = weights.sum(dim=-1, keepdim=True)            

    return x_restore, counts


def token_injector(x: torch.Tensor, x_group):

    B, _, C = x.shape

    # x = x.view(B, H * W, C)

    x_group = x_group.flatten((1))

    # separate the tokens into the three groups; rep, unique and normal
    x_rep = x[ x_group == 0 ].view (B , -1 , C)
    x_unique = x[ x_group == 1 ].view (B , -1 , C)
    x_normal = x[ x_group == 2 ].view (B , -1 , C)

    # merge rep and normal tokens
    values, counts, idx = inject_q_kv(
        q=x_rep, kv=x_normal, return_idx=True
    )

    kv_count = torch.gather(counts, 1, idx).unsqueeze(-1)

    refined_rep = (x_rep + values ) / (kv_count + 1)
    x_reduced = torch.cat([refined_rep, x_unique], dim=1)

    # computed the distance matrix between the reduced tokens (rep + normal) and the original tokens 
    # (the very input to the current transformer layer, no transformation)
    # qkv_dist_matrix = compute_distance(
    #     q=x, kv=x_reduced, mode='cosine'
    # )

    return x_reduced


def token_ejector(x_reduced: torch.Tensor, dist_mat: torch.Tensor, thres: float):
   
    x_restore, counts = ejector_q_kv(
        x_reduced=x_reduced,
        dist=dist_mat, 
        threshold_value=thres
    )

    return x_restore / (counts + 1e-10)



if __name__ == '__main__':
    B, H, W, C = 16, 14, 14, 768
    x = torch.randn(B, H, W, C, dtype=torch.float32)
 
    reduced_tokens = token_injector(x) # fuse the rep and normal tokens 

    x_and_kept_dist_matrix = compute_distance(
        q=x.view(B, H * W, C), kv=reduced_tokens, mode='cosine'
    )
    print(f'\nOriginal tokens: {x.view(B, H*W, C).shape}')
    print(f'Reduced tokens: {reduced_tokens.shape}')
    print(f'Distance matrix: {x_and_kept_dist_matrix.shape}')

    x_restore = token_ejector(reduced_tokens, x_and_kept_dist_matrix, 0.5) # restore the tokens using the distance matix from earlier
    print(f'\nRestored tokens: {x_restore.shape}')