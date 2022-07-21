from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import import_class

from .gcn_attention import gcn_unit_attention
from .net import Unit2D
from .temporal_transformer import TcnUnitAttention
from .temporal_transformer_windowed import TcnUnitAttentionBlock
from .unit_agcn import UnitAGCN
from .unit_gcn import unit_gcn

default_backbone_all_layers = [
    (3, 64, 1),
    (64, 64, 1),
    (64, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 256, 1),
]

default_backbone = [
    (64, 64, 1),
    (64, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 256, 1),
]


class Model(nn.Module):
    """Spatial temporal graph convolutional networks
                        for skeleton-based action recognition.

    Input shape:
        Input shape should be (N, C, T, V, M)
        where N is the number of samples,
              C is the number of input channels,
              T is the length of the sequence,
              V is the number of joints or graph nodes,
          and M is the number of people.

    Arguments:
        About shape:
            channel (int): Number of channels in the input data
            num_class (int): Number of classes for classification
            window_size (int): Length of input sequence
            num_point (int): Number of joints or graph nodes
            num_person (int): Number of people
        About net:
            use_data_bn: If true, the data will first input to a batch normalization layer
            backbone_config: The structure of backbone networks
        About graph convolution:
            graph: The graph of skeleton, represtented by a adjacency matrix
            graph_args: The arguments of graph
            mask_learning: If true, use mask matrixes to reweight the adjacency matrixes
            use_local_bn: If true, each node in the graph have specific parameters of batch normalzation layer
        About temporal convolution:
            multiscale: If true, use multi-scale temporal convolution
            temporal_kernel_size: The kernel size of temporal convolution
            dropout: The drop out rate of the dropout layer in front of each temporal convolution layer

    """

    def __init__(
        self,
        channel,
        num_class,
        window_size,
        num_point,
        attention,
        only_attention,
        tcn_attention,
        only_temporal_attention,
        attention_3,
        relative,
        double_channel,
        drop_connect,
        concat_original,
        dv,
        dk,
        Nh,
        dim_block1,
        dim_block2,
        dim_block3,
        all_layers,
        data_normalization,
        visualization,
        skip_conn,
        adjacency,
        bn_flag,
        weight_matrix,
        n,
        more_channels,
        num_person=1,
        use_data_bn=False,
        backbone_config=None,
        graph=None,
        graph_args: dict = None,
        mask_learning=False,
        use_local_bn=False,
        multiscale=False,
        kernel_temporal=9,
        dropout=0.5,
        agcn: bool = True,
        loss_fn: str = "cce",
    ):
        super().__init__()
        if graph is None:
            raise ValueError()
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        self.A = torch.from_numpy(self.graph.A)
        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.multiscale = multiscale
        self.attention = attention
        self.tcn_attention = tcn_attention
        self.drop_connect = drop_connect
        self.more_channels = more_channels
        self.concat_original = concat_original
        self.all_layers = all_layers
        self.dv = dv
        self.num = n
        self.Nh = Nh
        self.dk = dk
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.visualization = visualization
        self.double_channel = double_channel
        self.adjacency = adjacency

        # Different bodies share batchNorm parameters or not
        self.M_dim_bn = True
        if self.M_dim_bn:
            self.data_bn = nn.BatchNorm1d(channel * num_point * num_person)
        else:
            self.data_bn = nn.BatchNorm1d(channel * num_point)

        if self.all_layers and not self.double_channel:
            self.starting_ch = 64
        elif self.all_layers or not self.double_channel:
            self.starting_ch = 128
        else:
            self.starting_ch = 256

        kwargs = dict(
            A=self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn,
            dropout=dropout,
            kernel_size=kernel_temporal,
            attention=attention,
            only_attention=only_attention,
            tcn_attention=tcn_attention,
            only_temporal_attention=only_temporal_attention,
            attention_3=attention_3,
            relative=relative,
            weight_matrix=weight_matrix,
            more_channels=self.more_channels,
            drop_connect=self.drop_connect,
            data_normalization=self.data_normalization,
            skip_conn=self.skip_conn,
            adjacency=self.adjacency,
            starting_ch=self.starting_ch,
            visualization=self.visualization,
            all_layers=self.all_layers,
            dv=self.dv,
            dk=self.dk,
            Nh=self.Nh,
            num=n,
            dim_block1=dim_block1,
            dim_block2=dim_block2,
            dim_block3=dim_block3,
            num_point=num_point,
            agcn=agcn,
        )

        unit = TCN_GCN_unit_multiscale if self.multiscale else TCN_GCN_unit

        # backbone
        if backbone_config is None:
            if self.all_layers:
                backbone_config = default_backbone_all_layers
            else:
                backbone_config = default_backbone
        self.backbone = nn.ModuleList(
            [
                unit(in_c, out_c, stride=stride, **kwargs)
                for in_c, out_c, stride in backbone_config
            ]
        )
        if self.double_channel:
            backbone_in_c = backbone_config[0][0] * 2
            backbone_out_c = backbone_config[-1][1] * 2
        else:
            backbone_in_c = backbone_config[0][0]
            backbone_out_c = backbone_config[-1][1]
        backbone_out_t = window_size
        backbone = []
        for i, (in_c, out_c, stride) in enumerate(backbone_config):
            if self.double_channel:
                in_c = in_c * 2
                out_c = out_c * 2
            if i == 3 and concat_original:
                backbone.append(
                    unit(
                        in_c + channel,
                        out_c,
                        stride=stride,
                        last=i == len(default_backbone) - 1,
                        last_graph=(i == len(default_backbone) - 1),
                        layer=i,
                        **kwargs
                    )
                )
            else:
                backbone.append(
                    unit(
                        in_c,
                        out_c,
                        stride=stride,
                        last=i == len(default_backbone) - 1,
                        last_graph=(i == len(default_backbone) - 1),
                        layer=i,
                        **kwargs
                    )
                )
            if backbone_out_t % stride == 0:
                backbone_out_t = backbone_out_t // stride
            else:
                backbone_out_t = backbone_out_t // stride + 1
        self.backbone = nn.ModuleList(backbone)

        # head
        if not all_layers:
            self.gcn0 = (
                UnitAGCN(
                    channel,
                    backbone_in_c,
                    self.A,
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn,
                )
                if agcn
                else unit_gcn(
                    channel,
                    backbone_in_c,
                    self.A,
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn,
                )
            )

            self.tcn0 = Unit2D(
                backbone_in_c, backbone_in_c, kernel_size=kernel_temporal
            )

        # tail
        self.person_bn = nn.BatchNorm1d(backbone_out_c)
        self.gap_size = backbone_out_t
        if type(self.num_class) == list:
            self.fcn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(backbone_out_c, num_classes, kernel_size=1),
                        nn.Sigmoid(),
                    )
                    if loss_fn == "multilabel"
                    else nn.Conv1d(backbone_out_c, num_classes, kernel_size=1)
                    for num_classes in self.num_class
                ]
            )
        else:
            self.fcn = nn.Conv1d(backbone_out_c, self.num_class, kernel_size=1)

    def forward(self, x: torch.Tensor):
        N, C, T, V, M = x.size()
        if self.concat_original:
            x_coord = x
            x_coord = x_coord.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)

        # data bn
        time_mask = (torch.sum(x, dim=(1, 3, 4), keepdim=True) == 0).to(x.dtype)
        # from (N, 1, T, 1, 1) to (N*1, 1, T, 1)
        time_mask = (
            time_mask.permute(0, 4, 1, 2, 3)
            .contiguous()
            .view(N, 1, T, 1)
            .tile(1, 1, 1, V)
        )
        # N, 1, T, V, ---> N, V, 1, T  --> (N * V, 1, T)
        time_mask = time_mask.permute(0, 3, 1, 2).reshape(-1, 1, T)
        # time_mask = None

        space_mask = (torch.sum(x, dim=(1, 2, 4), keepdim=True) == 0).to(x.dtype)
        # from (N, 1, 1, V, 1) to (N*1, V, V)
        space_mask = (
            space_mask.permute(0, 4, 1, 2, 3).contiguous().view(N, V, 1).tile((1, 1, V))
        )
        if self.use_data_bn:
            if self.M_dim_bn:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
            x = self.data_bn(x)
            # to (N*M, C, T, V)
            x = (
                x.view(N, M, V, C, T)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
                .view(N * M, C, T, V)
            )
        else:
            # from (N, C, T, V, M) to (N*M, C, T, V)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # model
        if not self.all_layers:
            x = self.gcn0(x, space_mask)
            x = self.tcn0(x)

        for i, m in enumerate(self.backbone):
            if i == 3 and self.concat_original:
                x, time_mask = m(torch.cat((x, x_coord), dim=1), space_mask, time_mask)
            else:
                x, time_mask = m(x, space_mask, time_mask)

        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V))

        # M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)

        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2])

        # C fcn
        if type(self.num_class) == list:
            return [fcn(x).squeeze(-1) for fcn in self.fcn]
        return self.fcn(x).squeeze(-1)


class TCN_GCN_unit(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        A,
        attention,
        only_attention,
        tcn_attention,
        only_temporal_attention,
        relative,
        attention_3,
        dv,
        dk,
        Nh,
        num,
        dim_block1,
        dim_block2,
        dim_block3,
        num_point,
        weight_matrix,
        more_channels,
        drop_connect,
        starting_ch,
        all_layers,
        adjacency,
        data_normalization,
        visualization,
        skip_conn,
        layer=0,
        kernel_size=9,
        stride=1,
        dropout=0.5,
        use_local_bn=False,
        mask_learning=False,
        last=False,
        last_graph=False,
        agcn=False,
    ):
        super().__init__()
        self.A = A

        self.V = A.shape[-1]
        self.C = in_channel
        self.last = last
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.num_point = num_point
        self.adjacency = adjacency
        self.last_graph = last_graph
        self.layer = layer
        self.stride = stride
        self.drop_connect = drop_connect
        self.visualization = visualization
        self.all_layers = all_layers
        self.more_channels = more_channels

        if out_channel >= starting_ch and attention or (self.all_layers and attention):

            self.gcn1 = gcn_unit_attention(
                in_channel,
                out_channel,
                dv_factor=dv,
                dk_factor=dk,
                Nh=Nh,
                complete=True,
                relative=relative,
                only_attention=only_attention,
                layer=layer,
                incidence=A,
                bn_flag=True,
                last_graph=self.last_graph,
                more_channels=self.more_channels,
                drop_connect=self.drop_connect,
                adjacency=self.adjacency,
                num=num,
                data_normalization=self.data_normalization,
                skip_conn=self.skip_conn,
                visualization=self.visualization,
                num_point=self.num_point,
            )
        else:
            args = (in_channel, out_channel, A)
            kwargs = {"use_local_bn": use_local_bn, "mask_learning": mask_learning}
            self.gcn1 = UnitAGCN(*args, **kwargs) if agcn else unit_gcn(*args, **kwargs)

        if (
            out_channel >= starting_ch
            and tcn_attention
            or (self.all_layers and tcn_attention)
        ):
            if out_channel <= starting_ch and self.all_layers:
                self.tcn1 = TcnUnitAttentionBlock(
                    out_channel,
                    out_channel,
                    dv_factor=dv,
                    dk_factor=dk,
                    Nh=Nh,
                    relative=relative,
                    only_temporal_attention=only_temporal_attention,
                    dropout=dropout,
                    kernel_size_temporal=kernel_size,
                    stride=stride,
                    weight_matrix=weight_matrix,
                    bn_flag=True,
                    last=self.last,
                    layer=layer,
                    more_channels=self.more_channels,
                    drop_connect=self.drop_connect,
                    n=num,
                    data_normalization=self.data_normalization,
                    skip_conn=self.skip_conn,
                    visualization=self.visualization,
                    dim_block1=dim_block1,
                    dim_block2=dim_block2,
                    dim_block3=dim_block3,
                    num_point=self.num_point,
                )
            else:
                self.tcn1 = TcnUnitAttention(
                    out_channel,
                    out_channel,
                    dv_factor=dv,
                    dk_factor=dk,
                    Nh=Nh,
                    relative=relative,
                    only_temporal_attention=only_temporal_attention,
                    dropout=dropout,
                    kernel_size_temporal=kernel_size,
                    stride=stride,
                    weight_matrix=weight_matrix,
                    bn_flag=True,
                    last=self.last,
                    layer=layer,
                    more_channels=self.more_channels,
                    drop_connect=self.drop_connect,
                    n=num,
                    data_normalization=self.data_normalization,
                    skip_conn=self.skip_conn,
                    visualization=self.visualization,
                    num_point=self.num_point,
                )

        else:
            self.tcn1 = Unit2D(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                dropout=dropout,
                stride=stride,
            )
        if (in_channel != out_channel) or (stride != 1):
            self.down1 = Unit2D(in_channel, out_channel, kernel_size=1, stride=stride)
            self.mask_downsize = torch.nn.AvgPool1d(kernel_size=2, stride=2)
        else:
            self.down1 = None

    def forward(
        self,
        x: torch.Tensor,
        space_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # N, C, T, V = x.size()
        if self.down1 is not None and time_mask is not None:
            time_mask = (self.mask_downsize(time_mask) == 0).to(x.dtype)
        x = self.tcn1(self.gcn1(x, space_mask), time_mask) + (
            x if self.down1 is None else self.down1(x)
        )
        if x.size(-2) - time_mask.size(-1) == 1 and time_mask is not None:
            time_mask = F.pad(time_mask, (1, 0), "constant", 0)
        return x, time_mask


class TCN_GCN_unit_multiscale(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1, **kwargs):
        super().__init__()
        self.unit_1 = TCN_GCN_unit(
            in_channels,
            out_channels // 2,
            A,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs
        )
        self.unit_2 = TCN_GCN_unit(
            in_channels,
            out_channels - out_channels // 2,
            A,
            kernel_size=kernel_size * 2 - 1,
            stride=stride,
            **kwargs
        )

    def forward(
        self, x: torch.Tensor, space_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.cat(
            (self.unit_1(x, space_mask), self.unit_2(x, space_mask)), dim=1
        )
