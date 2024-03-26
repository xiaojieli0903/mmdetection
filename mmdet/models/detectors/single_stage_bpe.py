# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union, Optional, Dict
import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList, DetDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.utils import unpack_gt_instances
from .base import BaseDetector

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class SingleStageDetectorBPE(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # For loss_bpe
        self.loss_mse = nn.MSELoss()

        print(self.backbone, self.neck, self.bbox_head)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                idx_layer: Optional[int] = 0,
                idx_block: Optional[int] = 0,
                delta_x: Optional[torch.Tensor] = None,
                reforward: bool = False,
                images_origin: Optional[
                    torch.Tensor] = None) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'stem':
            return self.extract_stem(inputs, data_samples)
        elif mode == 'layer':
            return self.extract_layer(inputs, data_samples, idx_layer,
                                      delta_x, reforward, images_origin)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def calculate_reg_loss(self, delta_x, delta_y, loss_type='mse'):
        if loss_type == 'mse':
            delta_x_norm = torch.flatten(delta_x, 1)
            delta_x_norm = delta_x_norm / torch.sqrt(
                torch.sum(delta_x_norm ** 2, dim=1,
                          keepdims=True))
            delta_y_norm = torch.flatten(delta_y, 1)
            delta_y_norm = delta_y_norm / torch.sqrt(
                torch.sum(delta_y_norm ** 2, dim=1,
                          keepdims=True))
            return self.loss_mse(delta_y_norm, delta_x_norm)
        elif loss_type == 'cosine':
            similarity = torch.einsum(
                "nc,nc->n", [F.normalize(torch.flatten(delta_x, 1), dim=-1),
                             F.normalize(torch.flatten(delta_y, 1), dim=-1)])
            return -similarity.mean()

    def extract_stem(self, inputs: torch.Tensor,
                     data_samples: SampleList) -> list:
        out, out_cls = self.backbone.forward_stem(inputs)
        use_bpe_reg = self.backbone.use_bpe_reg
        nobp_type = self.backbone.nobp_type
        if nobp_type != 'layer_mergestem':
            losses = self.decode_head.loss_by_feat(out_cls, data_samples)
        else:
            losses = None

        if use_bpe_reg and nobp_type != 'layer_mergestem':
            delta_x = \
            torch.autograd.grad(outputs=losses['loss_ce'], inputs=out,
                                retain_graph=True)[0].detach()
        else:
            delta_x = None
        return [out, delta_x, losses]

    def extract_layer(self, inputs, data_samples=None, idx_layer=0,
                      delta_x=None, reforward=False, images_origin=None):
        if idx_layer == (len(self.backbone.stage_blocks) - 1):
            is_final = True
        else:
            is_final = False
        out, out_cls = self.backbone.forward_layer(inputs, idx_layer, data_samples=data_samples)

        out_recon = None
        if isinstance(out_cls, list):
            out_cls, out_recon = out_cls

        use_bpe_reg = self.backbone.use_bpe_reg
        use_reforward = self.backbone.use_reforward
        loss_weight_bpe = self.backbone.loss_weight_bpe if not is_final else self.backbone.loss_weight_bpe_final
        loss_weight_task = self.backbone.loss_weight_task if not is_final else self.backbone.loss_weight_task_final
        loss_type_bpe = self.backbone.loss_type_bpe

        if not is_final:
            losses = out_cls
        else:
            losses = self.bbox_head.loss(self.neck(self.backbone.feats_middle), data_samples)
        for key in losses:
            if isinstance(losses[key], list):
                for i in range(len(losses[key])):
                    losses[key][i] = loss_weight_task * losses[key][i]
            else:
                losses[key] = loss_weight_task * losses[key]

        if use_bpe_reg:
            if not reforward and delta_x is not None:
                if loss_weight_bpe != 0:
                    delta_x_bbox, delta_x_cls = delta_x
                    delta_y_bbox = torch.autograd.grad(outputs=losses['loss_bbox'],
                                                       inputs=inputs,
                                                       create_graph=True,
                                                       retain_graph=True)[0]
                    # delta_y_cls = torch.autograd.grad(outputs=losses['loss_cls'],
                    #                                    inputs=inputs,
                    #                                    create_graph=True,
                    #                                    retain_graph=True)[0]
                    losses[
                        'loss_bpe_bbox'] = loss_weight_bpe * self.calculate_reg_loss(
                        delta_x_bbox, delta_y_bbox, loss_type_bpe)
                    # losses[
                    #     'loss_bpe_cls'] = loss_weight_bpe * self.calculate_reg_loss(
                    #     delta_x_cls, delta_y_cls, loss_type_bpe)
            if ((not use_reforward) or (
                    use_reforward and reforward)) and not is_final and idx_layer == (len(self.backbone.stage_blocks) - 2):
                delta_x_bbox = torch.autograd.grad(outputs=losses['loss_bbox'],
                                                   inputs=out,
                                                   retain_graph=True)[0].detach()
                # delta_x_cls = torch.autograd.grad(outputs=losses['loss_cls'],
                #                               inputs=out,
                #                               retain_graph=True)[0].detach()
                delta_x_cls = None
                delta_x = [delta_x_bbox, delta_x_cls]
            else:
                delta_x = None
        else:
            delta_x = None

        return [out, delta_x, losses]
