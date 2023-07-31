import torch
from torch import nn

# from modeling_fastreid.layers import *
# from modeling_fastreid.modeling.backbones import build_backbone
# from modeling_fastreid.modeling.heads import build_reid_heads
# from modeling_fastreid.modeling.losses import *

# from fastreid.layers import *
# from fastreid.modeling.backbones import build_backbone
# from fastreid.modeling.heads import build_reid_heads
# from fastreid.modeling.losses import *
# from .build import META_ARCH_REGISTRY

from .resnet import build_resnet_backbone
from .layers import *
from .hrcn_head import HRCNHead
from .cross_entroy_loss import CrossEntropyLoss
import torch.nn.functional as F
# from metric_loss import TripletLoss
from layers.triplet_loss import TripletLoss

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input

# CrossEntropyLoss = F.cross_entropy
class Baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self._cfg = cfg

        # backbone
        self.backbone = build_resnet_backbone()

        # head
        pool_type = "avgpool"
        if pool_type == 'fastavgpool':
            pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':
            pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool":
            pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":
            pool_layer = Identity()
        else:
            raise KeyError("not supported pooling type for {}".format(pool_type))

        in_feat = 2048
        self.num_classes = num_classes
        self.circle_pool_layer = CenterPool(4,
                                            False,
                                            'cuda',
                                            'circle',
                                            True)
        self.heads = HRCNHead(None, in_feat, num_classes, pool_layer, 4, None)
        # self.heads = build_reid_heads(cfg, in_feat, num_classes, pool_layer, cfg.MODEL.HEADS.CENTER_NUM, None)


    def forward(self, images):
        # images = self.preprocess_image(batched_inputs)
        features, features_list = self.backbone(images)

        if self.training:
            # assert "targets" in batched_inputs, "Vehicle ID annotation are missing in training!"
            # targets = batched_inputs["targets"].long().to(self.device)

            # if targets.sum() < 0: targets.zero_()

            part_features_list = self.circle_pool_layer(features)
            all_features_list = features_list + part_features_list
            total_cls_outputs, total_pred_class_logits, total_features = self.heads(all_features_list, None)

            return [total_cls_outputs, total_pred_class_logits, total_features]

        else:
            part_features_list = self.circle_pool_layer(features)
            all_features_list = features_list + part_features_list
            total_features = self.heads(all_features_list)
            global_features = total_features[-1]

            return global_features

    # def preprocess_image(self, batched_inputs):
    #     """
    #     Normalize and batch the input images.
    #     """
    #     if isinstance(batched_inputs, dict):
    #         images = batched_inputs["images"].to(self.device)
    #     elif isinstance(batched_inputs, torch.Tensor):
    #         images = batched_inputs.to(self.device)
    #     images.sub_(self.pixel_mean).div_(self.pixel_std)
    #     return images

    def losses(self, outputs, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        cls_outputs, pred_class_logits, pred_features = outputs
        loss_dict = {}
        loss_names = ["CrossEntropyLoss", "TripletLoss"]

        # Log prediction accuracy
        # for i, pred_class_logit in enumerate(pred_class_logits):
        #     CrossEntropyLoss.log_accuracy(pred_class_logit.detach(), gt_labels, name='{}'.format(i))

        if "CrossEntropyLoss" in loss_names:
            for i, outputs in enumerate(cls_outputs):
                loss_dict['loss_cls_{}'.format(i)] = CrossEntropyLoss(self.num_classes)(outputs, gt_labels)

        if "TripletLoss" in loss_names:
            for i, features in enumerate(pred_features):
                loss_dict['loss_triplet_{}'.format(i)] = TripletLoss(0.0)(features, gt_labels)[0]

        return list(loss_dict.values())

class MyNet(nn.Module):
    def __init__(self, num_classes):
        super(MyNet, self).__init__()
        self.net1 = Baseline(num_classes)
        self.net2 = Baseline(num_classes)
        # self.net3 = Baseline(num_classes)
    
    def forward(self, x, label=None):
        if self.training:
            outputs_0 = self.net1(x[0])
            outputs_1 = self.net2(x[1])
            # outputs_2 = self.net3(x[2])

            loss_0 = self.net1.losses(outputs_0, label)
            loss_1 = self.net2.losses(outputs_1, label)
            # loss_2 = self.net3.losses(outputs_2, label)
            # import pdb; pdb.set_trace()
            # return [sum(outputs_0[1]), sum(outputs_1[1]), sum(outputs_2[1])], [sum(loss_0), sum(loss_1), sum(loss_2)]
            return [sum(outputs_0[1]), sum(outputs_1[1])], [sum(loss_0), sum(loss_1)]
        else:
            outputs_0 = self.net1(x[0])
            outputs_1 = self.net2(x[1])
            # outputs_2 = self.net3(x[2])
        
            # return [outputs_0, outputs_1, outputs_2]
            return [outputs_0, outputs_1]