import torch
import torch.nn as nn

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def param_dist_comput(f_x, f_y, weights):
    l = len(f_x)
    i = 0
    final_dist = 0
    for fx in f_x:
        for fy in f_y:
            dist = euclidean_dist(fx, fy)
            final_dist += weights[i] * dist
            i += 1
    return final_dist

def weighted_dist_comp(f_x, f_y, w_x, w_y):
    """
        f_x is a list for feature with shape [m, d], same as f_y
        w_x is a list for weights with shape [m, 1], same as w_y
    """
    w_x = [w.squeeze().unsqueeze(1) for w in w_x]
    w_y = [w.squeeze().unsqueeze(1) for w in w_y]
    l = len(f_x)
    total_w = 0
    for i in range(l):
        total_w += torch.mm(w_x[i], w_y[i].t())
    w_mat = []
    final_dist = 0
    for i in range(l):
        w = torch.mm(w_x[i], w_y[i].t()) / total_w
        dist = euclidean_dist(f_x[i], f_y[i])
        final_dist += dist * w
    
    return final_dist
    

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class WeightedTripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feats, weights, labels, normalize_feature=False):
        if normalize_feature:
            for i in len(global_feats):
                global_feats[i] = normalize(global_feats[i], axis=-1)
            # global_feat = normalize(global_feat, axis=-1)
        dist_mat = weighted_dist_comp(global_feats, global_feats, weights, weights)
        # dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class ParamTripletLoss(nn.Module):

    def __init__(self, N_branch=3, margin=None, use_gpu=True):
        super(ParamTripletLoss, self).__init__()
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

        self.N_branch = N_branch
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.weights = nn.Parameter(torch.randn(N_branch**2).cuda())
            self.softmax = nn.Softmax(dim=-1).cuda()
        else:
            self.weights = nn.Parameter(torch.randn(N_branch**2))
            self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, f_list, labels, normalize_feature=False):
        if normalize_feature:
            f_list = [normalize(f) for f in f_list]
        dist_mat_list = []
        for f1 in f_list:
            for f2 in f_list:
                dist_mat_list.append(euclidean_dist(f1, f2))

        # normlization
        min_w = torch.min(self.weights)
        max_w = torch.max(self.weights)
        normed_weights = (self.weights - min_w) / (max_w - min_w + 1e-12)
        
        dist_mat_list = [dist_mat * normed_weights[i] for i, dist_mat in enumerate(dist_mat_list)]
        dist_map = sum(dist_mat_list)
        dist_ap, dist_an = hard_example_mining(dist_map, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
        