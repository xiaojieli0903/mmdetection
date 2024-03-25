# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.model import normal_init
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 is_final=False,
                 classifier_cfg=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

        self.classifier_cfg = classifier_cfg
        self.is_final = is_final
        if self.classifier_cfg is not None:
            self.use_bpe_reg = self.classifier_cfg.get('use_bpe_reg', True)
            self.use_reforward = self.classifier_cfg.get('use_reforward',
                                                         True)
            self.inference_eval = self.classifier_cfg.get('inference_eval',
                                                          False)
            self.middle_eval = self.classifier_cfg.get('middle_eval',
                                                       True)
            self.pooling_size = self.classifier_cfg.get('pooling_size',
                                                        1)
            self.final_cls = self.classifier_cfg.get('final_cls', False)
            self.classifier_type = self.classifier_cfg.get('classifier_type',
                                                           'v1')
            self.loss_weight_infopro = self.classifier_cfg.get(
                'loss_weight_infopro', 0)

            if self.classifier_cfg is not None:
                self.auxiliary_classifier = MODELS.build(
                    self.classifier_cfg.bbox_head)

                if self.loss_weight_infopro > 0 and not is_final:
                    self.decoder = Decoder(out_channels)
                else:
                    self.decoder = None

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x, _, data_samples = x

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        # Auxiliary classifier operations
        def apply_auxiliary_classifier(x):
            return self.auxiliary_classifier.loss((x,), data_samples)

        # Check conditions for auxiliary classifier
        if self.classifier_cfg is not None:
            if self.training:
                # Auxiliary classifier in training mode
                if self.middle_eval and not self.is_final:
                    self.auxiliary_classifier.eval()
                if self.loss_weight_infopro > 0 and self.decoder is not None:
                    out_recon = self.decoder(out)
                else:
                    out_recon = None
                return [out, [apply_auxiliary_classifier(out), out_recon]]
            else:
                # Auxiliary classifier in evaluation mode
                if not self.is_final:
                    return out
                return out
        else:
            return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 is_final=False,
                 classifier_cfg=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        self.classifier_cfg = classifier_cfg
        self.is_final = is_final
        if self.classifier_cfg is not None:
            self.use_bpe_reg = self.classifier_cfg.get('use_bpe_reg', True)
            self.use_reforward = self.classifier_cfg.get('use_reforward',
                                                         True)
            self.inference_eval = self.classifier_cfg.get('inference_eval',
                                                          False)
            self.middle_eval = self.classifier_cfg.get('middle_eval',
                                                       True)
            self.pooling_size = self.classifier_cfg.get('pooling_size',
                                                        1)
            self.final_cls = self.classifier_cfg.get('final_cls', False)
            self.classifier_type = self.classifier_cfg.get('classifier_type',
                                                           'v1')
            self.loss_weight_infopro = self.classifier_cfg.get(
                'loss_weight_infopro', 0)

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)
        out_channels = planes * self.expansion

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)
            out_channels = self.after_conv3_plugins

        if self.classifier_cfg is not None:
            self.auxiliary_classifier = MODELS.build(
                self.classifier_cfg.bbox_head)
            # if self.final_cls and is_final:
            #     self.auxiliary_classifier = nn.Identity()
            # else:
            #     if self.classifier_type == 'v1':
                    # self.auxiliary_classifier = nn.Sequential()
                    # self.auxiliary_reg = nn.Sequential()
                    # in_channels = out_channels
                    # for i in range(2):
                    #     self.auxiliary_classifier.append(
                    #         ConvModule(
                    #             in_channels,
                    #             256,
                    #             3,
                    #             stride=1,
                    #             padding=1,
                    #             conv_cfg=self.conv_cfg,
                    #             norm_cfg=self.norm_cfg)
                    #     )
                    #     self.auxiliary_reg.append(
                    #         ConvModule(
                    #             in_channels,
                    #             256,
                    #             3,
                    #             stride=1,
                    #             padding=1,
                    #             conv_cfg=self.conv_cfg,
                    #             norm_cfg=self.norm_cfg)
                    #     )
                    #     in_channels = 256
                    # self.auxiliary_classifier.append(
                    #     nn.Conv2d(
                    #         in_channels,
                    #         720,
                    #         3,
                    #         padding=1)
                    # )
                    # self.auxiliary_reg.append(
                    #     nn.Conv2d(
                    #         in_channels,
                    #         36,
                    #         3,
                    #         padding=1)
                    # )

            if self.loss_weight_infopro > 0 and not is_final:
                self.decoder = Decoder(out_channels)
            else:
                self.decoder = None

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        if isinstance(x, list):
            x, _, data_samples = x

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        # Auxiliary classifier operations
        def apply_auxiliary_classifier(x):
            return self.auxiliary_classifier.loss((x, ), data_samples)

        # Check conditions for auxiliary classifier
        if self.classifier_cfg is not None:
            if self.training:
                # Auxiliary classifier in training mode
                if self.middle_eval and not self.is_final:
                    self.auxiliary_classifier.eval()
                if self.loss_weight_infopro > 0 and self.decoder is not None:
                    out_recon = self.decoder(out)
                else:
                    out_recon = None
                return [out, [apply_auxiliary_classifier(out), out_recon]]
            else:
                # Auxiliary classifier in evaluation mode
                if not self.is_final:
                    return out
                return out
        else:
            return out


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Defaults to 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Defaults to None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Defaults to dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Defaults to True
    """

    def __init__(self,
                 block: BaseModule,
                 inplanes: int,
                 planes: int,
                 num_blocks: int,
                 stride: int = 1,
                 avg_down: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 downsample_first: bool = True,
                 is_final=False,
                 **kwargs) -> None:
        self.block = block

        self.classifier_cfg = kwargs.pop('classifier_cfg', None)
        nobp_type = self.classifier_cfg.get('nobp_type',
                                            'block') if self.classifier_cfg else 'block'
        idx_layer = kwargs.pop('idx_layer', None)
        stages_classifier = kwargs.pop('stages_classifier', None)  # [1, 3]
        idx_middle_block = kwargs.pop('idx_middle_block', None)  # [2, 2]

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        if self.classifier_cfg is not None:
            import copy
            classifier_cfg = copy.deepcopy(self.classifier_cfg)
            classifier_cfg['bbox_head'] = self.classifier_cfg['bbox_head'][idx_layer]
        else:
            classifier_cfg = None
        if downsample_first:
            layers = []
            _has_classifier = classifier_cfg is not None and (
                    nobp_type in ['block'] or
                    ((idx_middle_block[idx_layer] == 0) and nobp_type in [
                        'layer',
                        'layer_mergestem']
                     ))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    classifier_cfg=classifier_cfg if _has_classifier else None,
                    is_final=(num_blocks == 1) and is_final,
                    **kwargs))
            inplanes = planes * block.expansion
            for i in range(1, num_blocks):
                if isinstance(idx_middle_block[idx_layer], list):
                    _has_classifier = classifier_cfg is not None and (
                            nobp_type in ['block'] or
                            (i in idx_middle_block[
                                idx_layer] and nobp_type in ['layer',
                                                             'layer_mergestem']
                             ))
                else:
                    _has_classifier = classifier_cfg is not None and (
                            nobp_type in ['block'] or
                            (i == idx_middle_block[
                                idx_layer] and nobp_type in [
                                 'layer',
                                 'layer_mergestem']
                             ))
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        is_final=(i == (num_blocks - 1)) and is_final,
                        classifier_cfg=classifier_cfg if _has_classifier else None,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for i in range(num_blocks - 1):
                if isinstance(idx_middle_block[idx_layer], list):
                    _has_classifier = classifier_cfg is not None and (
                            nobp_type in ['block'] or
                            (i in idx_middle_block[
                                idx_layer] and nobp_type in ['layer',
                                                             'layer_mergestem']
                             ))
                else:
                    _has_classifier = classifier_cfg is not None and (
                            nobp_type in ['block'] or
                            (i == idx_middle_block[
                                idx_layer] and nobp_type in [
                                 'layer',
                                 'layer_mergestem']
                             ))
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        is_final=(i == (num_blocks - 1)) and is_final,
                        classifier_cfg=classifier_cfg if _has_classifier else None,
                        **kwargs))
            if isinstance(idx_middle_block[idx_layer], list):
                _has_classifier = classifier_cfg is not None and (
                        nobp_type in ['block'] or
                        ((num_blocks - 1) in idx_middle_block[
                            idx_layer] and nobp_type in ['layer',
                                                         'layer_mergestem']
                         ))
            else:
                _has_classifier = classifier_cfg is not None and (
                        nobp_type in ['block'] or
                        ((num_blocks - 1) == idx_middle_block[
                            idx_layer] and nobp_type in [
                             'layer',
                             'layer_mergestem']
                         ))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    classifier_cfg=classifier_cfg if _has_classifier else None,
                    is_final=is_final,
                    **kwargs))
        super().__init__(*layers)


@MODELS.register_module()
class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     # print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 classifier_cfg=None):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self.classifier_cfg = classifier_cfg
        if self.classifier_cfg is not None:
            self.use_nobp = True
            self.nobp_type = self.classifier_cfg.get('nobp_type', 'block')
            self.use_bpe_reg = self.classifier_cfg.get('use_bpe_reg', True)
            self.use_reforward = self.classifier_cfg.get('use_reforward',
                                                         True)
            self.inference_eval = self.classifier_cfg.get('inference_eval',
                                                          False)
            self.middle_eval = self.classifier_cfg.get('middle_eval',
                                                       True)
            self.pooling_size = self.classifier_cfg.get('pooling_size',
                                                        1)
            self.final_cls = self.classifier_cfg.get('final_cls', False)
            self.loss_weight_bpe = self.classifier_cfg.get('loss_weight_bpe',
                                                           1.0)
            self.loss_weight_bpe_final = self.classifier_cfg.get(
                'loss_weight_bpe_final',
                self.loss_weight_bpe)
            self.classifier_type = self.classifier_cfg.get('classifier_type',
                                                           'v1')
            self.stages_classifier = self.classifier_cfg.get(
                'stages_classifier',
                range(len(self.stage_blocks)))
            self.idx_middle_block = self.classifier_cfg.get('idx_middle_block',
                                                            [num_block - 1 for
                                                             num_block in
                                                             self.stage_blocks])
            self.loss_weight_infopro = self.classifier_cfg.get(
                'loss_weight_infopro', 0)
            self.loss_weight_task = self.classifier_cfg.get('loss_weight_task',
                                                            1.0)
            self.loss_weight_task_final = self.classifier_cfg.get(
                'loss_weight_task_final',
                self.loss_weight_task)
            self.loss_type_bpe = self.classifier_cfg.get('loss_type_bpe',
                                                         'mse')
            self.update_classifier = self.classifier_cfg.get(
                'update_classifier', False)
            self.idx_stop = self.classifier_cfg.get('idx_stop', -1)
            self.reforward_mode = self.classifier_cfg.get('reforward_mode',
                                                          'classifier')
        else:
            self.classifier_cfg = {}
            self.use_nobp = False
            self.nobp_type = self.classifier_cfg.get('nobp_type', 'block')
            self.use_bpe_reg = self.classifier_cfg.get('use_bpe_reg', False)
            self.use_reforward = self.classifier_cfg.get('use_reforward',
                                                         False)
            self.inference_eval = self.classifier_cfg.get('inference_eval',
                                                          False)
            self.middle_eval = self.classifier_cfg.get('middle_eval',
                                                       False)
            self.pooling_size = self.classifier_cfg.get('pooling_size',
                                                        1)
            self.final_cls = self.classifier_cfg.get('final_cls', False)
            self.loss_weight_bpe = self.classifier_cfg.get('loss_weight_bpe',
                                                           1.0)
            self.loss_weight_bpe_final = self.classifier_cfg.get(
                'loss_weight_bpe_final',
                self.loss_weight_bpe)
            self.stages_classifier = self.classifier_cfg.get(
                'stages_classifier',
                range(len(self.stage_blocks)))
            self.idx_middle_block = self.classifier_cfg.get('idx_middle_block',
                                                            [num_block - 1 for
                                                             num_block in
                                                             self.stage_blocks])
            self.loss_weight_infopro = self.classifier_cfg.get(
                'loss_weight_infopro', 0)
            self.loss_weight_task = self.classifier_cfg.get('loss_weight_task',
                                                            1.0)
            self.loss_weight_task_final = self.classifier_cfg.get(
                'loss_weight_task_final',
                self.loss_weight_task)
            self.loss_type_bpe = self.classifier_cfg.get('loss_type_bpe',
                                                         'mse')
            self.update_classifier = self.classifier_cfg.get(
                'update_classifier', False)
            self.idx_stop = self.classifier_cfg.get('idx_stop', -1)
            self.reforward_mode = self.classifier_cfg.get('reforward_mode',
                                                          'classifier')
            self.classifier_cfg = None

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i

            _has_classifier = self.classifier_cfg is not None and (
                    (self.nobp_type in ['layer',
                                        'layer_mergestem']
                     and i in self.stages_classifier) or self.nobp_type == 'block')
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg,
                classifier_cfg=self.classifier_cfg if _has_classifier else None,
                is_final=False if i != (len(self.stage_blocks) - 1) else True,
                idx_layer=i,
                stages_classifier=self.stages_classifier,
                idx_middle_block=self.idx_middle_block
            )

            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feats_middle = [None for i in range(len(self.stage_blocks))]

        if self.classifier_cfg is not None and self.nobp_type in ['layer',
                                                                  'layer_mergestem']:
            idx_pre = 0
            stage_blocks_new = []
            for idx in self.stages_classifier:
                stage_blocks_new.append(
                    sum(self.stage_blocks[idx_pre:(idx + 1)]))
                idx_pre = idx + 1
            self.stage_blocks_origin = self.stage_blocks
            self.stage_blocks = stage_blocks_new

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def forward_stem(self, x):
        if self.deep_stem:
            out = self.stem(x)
        else:
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
        out = self.maxpool(out)

        if self.classifier_cfg is not None and self.training and self.nobp_type != 'layer_mergestem':
            if self.middle_eval:
                self.auxiliary_classifier.eval()
                self.auxiliary_classifier_linear.eval()
            out_cls = self.auxiliary_classifier(out)
            out_cls = self.auxiliary_classifier_linear(
                out_cls.view(out_cls.shape[0], -1))
            return [out, out_cls]
        else:
            return [out, None]

    def forward_layer(self, x, idx_layer=0, reforward=False, data_samples=None):
        if len(self.stage_blocks) == self.num_stages:
            module = getattr(self, f'layer{idx_layer + 1}')
            if isinstance(x, list):
                x.append(data_samples)
            else:
                x = [x, None, data_samples]
            x = module(x)
            self.feats_middle[idx_layer] = x.detach() if not isinstance(x, list) else x[0].detach()
            return x
        else:
            if not reforward or (reforward and self.reforward_mode == 'all'):
                idx_start = 0 if idx_layer < 1 else self.stages_classifier[
                                                        idx_layer - 1] + 1
                if idx_start >= len(self.stage_blocks_origin):
                    idx_start -= 1
                for idx in range(idx_start,
                                 self.stages_classifier[idx_layer] + 1):
                    num_blocks = self.stage_blocks_origin[idx]
                    num_blocks_last = self.stage_blocks_origin[
                        idx - 1] if idx > 0 else -1
                    idx_middle_block_last = self.idx_middle_block[
                        idx - 1] if idx > 0 else -1
                    idx_middle_block = self.idx_middle_block[idx]
                    if isinstance(idx_middle_block, list):
                        if idx_layer != len(self.stage_blocks) - 1:
                            if num_blocks_last > idx_middle_block_last + 1:
                                for idx_block in range(
                                        idx_middle_block_last + 1,
                                        num_blocks_last):
                                    if isinstance(x, list):
                                        x.append(data_samples)
                                    else:
                                        x = [x, None, data_samples]
                                    x = getattr(self, f'layer{idx}')[
                                        idx_block](x)
                                if isinstance(x, list):
                                    self.feats_middle[idx - 1] = x[0].detach() if idx in [1,2,3] else x[0] 
                                else:
                                    self.feats_middle[idx - 1] = x.detach() if idx in [1,2,3] else x 
                            for idx_block in range(0, idx_middle_block[0] + 1):
                                if isinstance(x, list):
                                    x.append(data_samples)
                                else:
                                    x = [x, None, data_samples]
                                x = getattr(self, f'layer{idx + 1}')[
                                    idx_block](x)
                                if idx_block == num_blocks - 1:
                                    if isinstance(x, list):
                                        self.feats_middle[idx] = x[0].detach() if idx in [0,1,2] else x[0] 
                                    else:
                                        self.feats_middle[idx] = x.detach() if idx in [0,1,2] else x 
                        else:
                            for idx_block in range(idx_middle_block[0] + 1,
                                                   idx_middle_block[1] + 1):
                                if isinstance(x, list):
                                    x.append(data_samples)
                                else:
                                    x = [x, None, data_samples]
                                x = getattr(self, f'layer{idx + 1}')[
                                    idx_block](x)
                                if idx_block == num_blocks - 1:
                                    if isinstance(x, list):
                                        self.feats_middle[idx] = x[0].detach() if idx in [0,1,2] else x[0] 
                                    else:
                                        self.feats_middle[idx] = x.detach() if idx in [0,1,2] else x 
                    else:
                        if num_blocks_last > idx_middle_block_last + 1:
                            for idx_block in range(idx_middle_block_last + 1,
                                                   num_blocks_last):
                                if isinstance(x, list):
                                    x.append(data_samples)
                                else:
                                    x = [x, None, data_samples]
                                x = getattr(self, f'layer{idx}')[idx_block](x)
                                if isinstance(x, list):
                                    self.feats_middle[idx - 1] = x[0].detach() if idx in [1,2,3] else x[0] 
                                else:
                                    self.feats_middle[idx - 1] = x.detach() if idx in [1,2,3] else x 
                            self.feats_middle[idx - 1] = x.detach() if idx in [1,2,3] else x 
                        for idx_block in range(0, idx_middle_block + 1):
                            if isinstance(x, list):
                                x.append(data_samples)
                            else:
                                x = [x, None, data_samples]
                            x = getattr(self, f'layer{idx + 1}')[idx_block](x)
                            if idx_block == num_blocks - 1:
                                if isinstance(x, list):
                                    self.feats_middle[idx] = x[0].detach() if idx in [0,1,2] else x[0] 
                                else:
                                    self.feats_middle[idx] = x.detach() if idx in [0,1,2] else x 
                return x
            elif reforward and self.reforward_mode == 'classifier':
                idx_middle_block = self.idx_middle_block[
                    self.stages_classifier[idx_layer]]
                x = getattr(self, f'layer{self.stages_classifier[idx_layer] + 1}')[
                    idx_middle_block](x, mode='classifier')
                return x

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
