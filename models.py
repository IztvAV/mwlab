import copy
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torchvision

import filters.mwfilter_lightning
from filters import MWFilter, CouplingMatrix
from filters.mwfilter_optim.base import FastMN2toSParamCalculation

_activations = {
        'relu': F.relu,
        'leaky_relu': F.leaky_relu,
        'elu': F.elu,
        'selu': F.selu,
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'swish': lambda x: x * torch.sigmoid(x),
        'mish': lambda x: x * torch.tanh(F.softplus(x)),
        'gelu': F.gelu,
        'none': lambda x: x,
        'rrelu': F.rrelu,
        'relu6': F.relu6,
        'soft_sign': F.softsign,
        'soft_plus': F.softplus,
        # --- Gaussian Error Linear Unit v2 ---
        'gelu_approx': lambda x: 0.5 * x * (
                    1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * x ** 3))),

        # --- Bent Identity ---
        'bent_identity': lambda x: ((torch.sqrt(x ** 2 + 1) - 1) / 2) + x,
        # --- Sinusoidal ---
        'sin': torch.sin,
        'cos': torch.cos,
        'sinc': lambda x: torch.where(x == 0, torch.ones_like(x), torch.sin(x) / x),
        # --- Snake (периодическая) ---
        'snake': lambda x, alpha=1.0: x + (1.0 / alpha) * torch.sin(alpha * x) ** 2,
        # --- SReLU (Simplified ReLU) ---
        'srelu': lambda x: torch.max(torch.min(x, torch.tensor(1.0)), torch.tensor(-1.0)),
        # --- TanhExp ---
        'tanh_exp': lambda x: x * torch.tanh(torch.exp(x)),
        # --- SQNL (Softsign Quartic Nonlinearity) ---
        'sqnl': lambda x: torch.where(x > 2, torch.ones_like(x),
                                      torch.where(x < -2, -torch.ones_like(x),
                                                  x - x ** 3 / 6)),
        # --- Gaussian ---
        'gaussian': lambda x: torch.exp(-x ** 2),
    }


def get_activation(name):
    """Возвращает функцию активации по имени"""
    return _activations.get(name.lower(), F.relu)

def get_available_activations():
    return list(_activations.keys())


class LSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=10, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        direction_mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_mult, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class GRU(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=10):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class BiRNN(nn.Module):
    def __init__(self, in_channels=8, hidden_size=64, num_layers=3, out_channels=10, droupout=0.2, rnn_type='lstm'):
        super().__init__()
        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU if rnn_type == 'gru' else nn.RNN
        self.rnn = rnn_class(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=droupout
        )
        self.fc = nn.Linear(hidden_size * 2, out_channels)
        self.hidden_size = hidden_size

    def forward(self, x):
        out, _ = self.rnn(x)
        # Конкатенируем прямое и обратное направления
        out = torch.cat([out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]], dim=1)
        return self.fc(out)


class Seq2Seq(nn.Module):
    def __init__(self, in_channels=8, hidden_size=64, out_channels=10):
        super().__init__()
        self.encoder = nn.LSTM(in_channels, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_channels)
        self.proj = nn.Linear(out_channels, hidden_size)  # Проекция обратно в hidden_size
        self.hidden_size = hidden_size

    def forward(self, x, target_len=1):
        # Кодирование
        _, (hidden, cell) = self.encoder(x)

        # Инициализация декодера
        decoder_input = torch.zeros(x.size(0), 1, self.hidden_size).to(x.device)
        outputs = []

        for _ in range(target_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            out = self.fc(out.squeeze(1))  # [batch, out_channels]
            outputs.append(out)

            # Проецируем выход обратно в пространство скрытого состояния
            decoder_input = self.proj(out.unsqueeze(1))  # [batch, 1, hidden_size]

        return torch.stack(outputs, dim=1)


class VGG1D(nn.Module):
    def __init__(self, in_channels=8, out_channels=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, out_channels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DenseLayer1D(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, activation='relu'):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.activation = get_activation(activation)
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)

    def forward(self, x):
        out = self.conv(self.activation(self.norm(x)))
        return out


class DenseBlock1D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer1D(channels, growth_rate, activation=activation))
            channels += growth_rate  # каждый слой добавляет новые каналы

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)


class ResNet1DFlexibleDense(nn.Module):
    def __init__(self, in_channels=8, out_channels=10,
                 first_conv_channels=64, first_conv_kernel=7,
                 dense_blocks_cfg=[(64, 32, 4), (128, 32, 4)],
                 activation_in='relu', activation_block='relu'):
        """
        dense_blocks_cfg: список кортежей (in_channels, growth_rate, num_layers)
        """
        super().__init__()
        self.activation_name = activation_in
        self.activation = get_activation(activation_in)

        self.conv1 = nn.Conv1d(in_channels, first_conv_channels,
                               kernel_size=first_conv_kernel, stride=2,
                               padding=first_conv_kernel // 2)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=first_conv_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Создаем Dense-блоки
        self.dense_blocks = nn.ModuleList()
        self.transition_blocks = nn.ModuleList()
        in_ch = first_conv_channels
        for out_ch, growth_rate, num_layers in dense_blocks_cfg:
            dense_block = DenseBlock1D(in_ch, growth_rate, num_layers,
                                       activation=activation_block)
            self.dense_blocks.append(dense_block)

            # После каждого DenseBlock делаем bottleneck 1x1 conv, чтобы привести к out_ch
            total_channels = in_ch + growth_rate * num_layers
            transition = nn.Conv1d(total_channels, out_ch, kernel_size=1, bias=False)
            self.transition_blocks.append(transition)

            in_ch = out_ch

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, out_channels)

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.maxpool(x)

        for dense_block, transition in zip(self.dense_blocks, self.transition_blocks):
            x = dense_block(x)
            x = transition(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, activation='relu', use_se=False, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

        # Выбор функции активации
        self.activation = get_activation(activation)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock1D(out_channels, reduction=se_reduction)
        self.dropout = nn.Dropout1d(p=0.1)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        out += self.shortcut(x)  # Residual connection
        out = self.activation(out)
        return out


class BottleneckBlock1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1,
                 kernel_size=3, activation='relu', use_se=False, se_reduction=16):
        super().__init__()
        mid_channels = out_channels // self.expansion
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.activation = get_activation(activation)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock1D(out_channels, reduction=se_reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.use_se:
            out = self.se(out)
        out += self.shortcut(x)
        return self.activation(out)



class ResNet1D(nn.Module):
    def __init__(self, in_channels=8, out_channels=10):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        self.layer5 = self.make_layer(512, 1024, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, out_channels)

    @staticmethod
    def make_layer(in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock1D(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# class ResNet1DFlexible(nn.Module):
#     def __init__(self, in_channels=8, out_channels=10,
#                  first_conv_channels=64, first_conv_kernel=7,
#                  first_maxpool_kernel=3,
#                  layer_channels=[64, 128, 256, 512],
#                  num_blocks=[1, 2, 3, 1],
#                  activation_in='relu', activation_block='relu',
#                  use_se=True, se_reduction=16):
#         super().__init__()
#
#         self.activation_name = activation_in
#         self.activation = get_activation(activation_in)
#
#         dilation = 1
#         self.conv1 = nn.Conv1d(in_channels, first_conv_channels,
#                                groups=1,
#                                kernel_size=first_conv_kernel,
#                                stride=2,
#                                padding=dilation*(first_conv_kernel // 2),
#                                dilation=dilation)
#         self.bn1 = nn.BatchNorm1d(first_conv_channels)
#         # self.bn1 = nn.GroupNorm(num_groups=8, num_channels=first_conv_channels)
#         self.maxpool = nn.MaxPool1d(kernel_size=first_maxpool_kernel, stride=2, padding=first_maxpool_kernel // 2)
#
#         self.layers = nn.ModuleList()
#         in_ch = first_conv_channels
#         for out_ch, n_blocks in zip(layer_channels, num_blocks):
#             self.layers.append(
#                 self.make_layer(in_ch, out_ch, n_blocks,
#                                 stride=2 if in_ch != out_ch else 1,
#                                 activation=activation_block,
#                                 use_se=use_se,
#                                 se_reduction=se_reduction)
#             )
#             in_ch = out_ch
#
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             # nn.LayerNorm(in_ch),
#             nn.Flatten(),
#             nn.Linear(in_ch, out_channels)
#         )
#         self.shortcut = nn.Sequential(
#             nn.BatchNorm1d(in_channels),
#             nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten()
#         )
#
#     @staticmethod
#     def make_layer(in_channels, out_channels, num_blocks, stride, activation, use_se, se_reduction):
#         layers = [BasicBlock1D(in_channels, out_channels, stride, activation,
#                                use_se=use_se, se_reduction=se_reduction)]
#         for _ in range(1, num_blocks):
#             layers.append(BasicBlock1D(out_channels, out_channels, stride=1,
#                                        activation=activation, use_se=use_se, se_reduction=se_reduction))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         identity = self.shortcut(x)
#         x = self.activation(self.bn1(self.conv1(x)))
#         x = self.maxpool(x)
#
#         for layer in self.layers:
#             x = layer(x)
#
#         x = self.avgpool(x)
#         # if type(self.fc) != nn.Identity:
#             # x = nn.Flatten(x)
#             # x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x += identity
#         return x


class ResNet1DFlexible(nn.Module):
    def __init__(self,
                 in_channels=8, out_channels=10,
                 first_conv_channels=64, first_conv_kernel=7,
                 first_maxpool_kernel=3,
                 block_kernel_size=3,
                 layer_channels=[64, 128, 256, 512],
                 num_blocks=[1, 2, 3, 1],
                 activation_in='relu', activation_block='relu',
                 use_se=True, se_reduction=16,
                 block_type='basic'  # <-- NEW
                 ):
        super().__init__()

        self.block_type = block_type.lower()
        self.activation_name = activation_in
        self.activation = get_activation(activation_in)

        dilation = 1
        self.conv1 = nn.Conv1d(in_channels, first_conv_channels,
                               groups=1,
                               kernel_size=first_conv_kernel,
                               stride=2,
                               padding=dilation * (first_conv_kernel // 2),
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(first_conv_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=first_maxpool_kernel, stride=2,
                                    padding=first_maxpool_kernel // 2)

        self.layers = nn.ModuleList()
        in_ch = first_conv_channels
        for out_ch, n_blocks in zip(layer_channels, num_blocks):
            self.layers.append(
                self.make_layer(in_ch, out_ch, n_blocks,
                                stride=2 if in_ch != out_ch else 1,
                                activation=activation_block,
                                use_se=use_se,
                                se_reduction=se_reduction,
                                block_kernel_size=block_kernel_size)
            )
            # Обновляем входные каналы с учётом expansion для bottleneck
            if self.block_type == 'bottleneck':
                in_ch = out_ch
            else:
                in_ch = out_ch

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def make_layer(self, in_channels, out_channels, num_blocks, stride, activation,
                   use_se, se_reduction, block_kernel_size):
        if self.block_type == 'bottleneck':
            block = BottleneckBlock1D
        else:
            block = BasicBlock1D

        layers = [block(in_channels, out_channels, stride=stride,
                        kernel_size=block_kernel_size, activation=activation,
                        use_se=use_se, se_reduction=se_reduction)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1,
                                kernel_size=block_kernel_size, activation=activation,
                                use_se=use_se, se_reduction=se_reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x += identity
        return x


class ModelWithCorrection(nn.Module):
    def __init__(self,
                 main_model: nn.Module, correction_model: nn.Module,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._main_model = main_model
        self._correction_model = correction_model

    def forward(self, x):
        main_output = self._main_model(x)
        correction = self._correction_model(main_output)
        final = main_output + correction

        # matrix = CouplingMatrix.from_factors(main_output, links=self._origin_filter.coupling_matrix.links,
        #                                      matrix_order=self._origin_filter.coupling_matrix.matrix_order,
        #                                      device=x.device)
        # _, s11_pred, s21_pred, s22_pred = self._fast_calc.BatchedRespM2(matrix, with_s22=True)
        # # Разделим на действительную и мнимую части
        # s11_re, s11_im = s11_pred.real, s11_pred.imag
        # s21_re, s21_im = s21_pred.real, s21_pred.imag
        # s22_re, s22_im = s22_pred.real, s22_pred.imag
        #
        # # Соберём в нужном порядке: [re..., im...]
        # s_parts = [s11_re, s21_re, s22_re, s11_im, s21_im, s22_im]
        # s_concat = torch.stack(s_parts, dim=1)  # shape: [batch_size, 6, freq_len]
        #
        # # scaled_decoded = self._scaler_in(s_concat)
        # decoded = self._correction_model(s_concat)
        return final


class ModelWithCorrectionAndSparameters(nn.Module):
    def __init__(self,
                 main_model: nn.Module, correction_model: nn.Module,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._main_model = main_model
        self._correction_model = correction_model

    def forward(self, x):
        main_output = self._main_model(x)
        correction = self._correction_model(x, main_output)
        final = correction
        return final



class CorrectEachFeature(nn.Module):
    def __init__(self,
                 main_model: nn.Module, correction_model: nn.Module, correct_features: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._main_model = main_model
        self._correction_models = [copy.deepcopy(correction_model).to("cuda") for _ in range(correct_features)]

    def forward(self, x):
        main_output = self._main_model(x)
        # Разбиваем на отдельные признаки
        corrected_features = []
        for i, corr_model in enumerate(self._correction_models):
            feature = main_output[:, i].unsqueeze(1)  # [batch_size, 1]
            correction = corr_model(feature)  # [batch_size, 1]
            corrected_feature = feature + correction  # [batch_size, 1]
            corrected_features.append(corrected_feature)

        # Собираем обратно
        final = torch.cat(corrected_features, dim=1)  # [batch_size, output_size]
        return final


class CorrectionTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, 30, 1)
        x = self.input_proj(x)  # (B, 30, d_model)
        x = x.permute(1, 0, 2)  # (30, B, d_model)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.output_proj(x).squeeze(-1)  # (B, 30)
        return x


class CorrectionMLP(nn.Module):
    def __init__(self, input_dim=30, hidden_dims=[64, 128, 64], output_dim=30, activation_fun='relu'):
        super().__init__()
        self.activation_fun = get_activation(activation_fun)

        # Сохраняем линейные слои в ModuleList
        self.in_activation = nn.ReLU()
        self.norm = nn.LayerNorm(input_dim)
        self.layers = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            self.normalizations.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        self.out_layer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, output_dim)
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, output_dim, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        identity = x
        for layer, norm in tuple(zip(self.layers, self.normalizations)):
            x = layer(x)
            x = norm(x)
            x = self.activation_fun(x)
        x = self.out_layer(x)
        x += identity
        return x


class CorrectionNet(nn.Module):
    def __init__(self, s_shape, m_dim, hidden_dim=256):
        super().__init__()
        s_dim = s_shape[0] * s_shape[1]  # 301 * 8 = 2408
        self.fc1 = nn.Linear(s_dim + m_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, m_dim)

    def forward(self, s_params, matrix_pred):

        s_params = s_params.view(s_params.size(0), -1)  # B x 2408
        x = torch.cat([s_params, matrix_pred], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        delta = self.fc3(x)
        return matrix_pred + delta


class CorrectionCNN1D(nn.Module):
    def __init__(self, input_len=30, output_dim=30):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.norm = nn.LayerNorm(16*input_len)
        self.fc = nn.Linear(16 * input_len, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 30)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)  # (B, 16*30)
        x = self.fc(self.norm(x))
        return x


class RobustAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        return self.dropout(out.unsqueeze(-1)) * x


class SEAttention(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(base_channels * 8, base_channels * 8 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(base_channels * 8 // 16, base_channels * 8, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.fc(x)
        return y


class ResNeXtBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4):
        super().__init__()
        width = int(out_channels * (base_width / 64)) * cardinality

        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)

        self.conv2 = nn.Conv1d(
            width, width, kernel_size=3,
            stride=stride, padding=1,
            groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm1d(width)

        self.conv3 = nn.Conv1d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x += residual
        return F.relu(x)


class ResNeXt1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, cardinality=32, base_width=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 1, stride=1, cardinality=cardinality, base_width=base_width)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, cardinality=cardinality, base_width=base_width)
        self.layer3 = self._make_layer(128, 256, 3, stride=2, cardinality=cardinality, base_width=base_width)
        self.layer4 = self._make_layer(256, 512, 1, stride=2, cardinality=cardinality, base_width=base_width)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, out_channels)

    def _make_layer(self, in_channels, out_channels, blocks, stride, cardinality, base_width):
        layers = [ResNeXtBlock1D(in_channels, out_channels, stride, cardinality, base_width)]
        for _ in range(1, blocks):
            layers.append(ResNeXtBlock1D(out_channels, out_channels, 1, cardinality, base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LeNet1D(nn.Module):
    def __init__(self, in_channels=8, num_points=301, out_channels=10):
        super().__init__()

        # Свёрточные слои (1D вместо 2D)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 6, kernel_size=5, padding=2),  # [B, 8, 301] -> [B, 6, 301]
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),  # -> [B, 6, 150]

            nn.Conv1d(6, 16, kernel_size=5, padding=2),  # -> [B, 16, 150]
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),  # -> [B, 16, 75]
        )

        # Автоматический расчёт размера перед полносвязными слоями
        self.flatten_size = self._get_flatten_size(in_channels, num_points)

        # Полносвязные слои
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_channels),
        )

    def _get_flatten_size(self, input_channels, num_points):
        """Вычисляет размер данных перед полносвязными слоями."""
        x = torch.zeros(1, input_channels, num_points)  # Пробный тензор
        x = self.features(x)
        return x.numel()  # Размер после flatten (batch_size=1)

    def forward(self, x):
        x = self.features(x)  # [B, C, L] -> [B, 16, 75]
        x = x.view(x.size(0), -1)  # Flatten: [B, 16*75]
        x = self.classifier(x)
        return x


class Simple_Opt_3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Simple_Opt_3, self).__init__()
        # Количество выходных аргументов
        self.nargout = 1
        # Количество выходных каналов

        # # --------------------------  1 conv-слой ---------------------------
        # self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding='same')
        # # --------------------------  2 conv-слой ---------------------------
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
        # # --------------------------  3 conv-слой ---------------------------
        # self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
        # # --------------------------  4 conv-слой ---------------------------
        # self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
        # self.seq_conv = nn.Sequential(
        #     # --------------------------  1 conv-слой ---------------------------
        #     self.conv1,
        #     nn.ReLU(),
        #     nn.AvgPool1d(kernel_size=3, stride=3, padding=1),
        #     # --------------------------  2 conv-слой ---------------------------
        #     self.conv2,
        #     nn.ReLU(),
        #     self.conv2,
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        #     # --------------------------  3 conv-слой ---------------------------
        #     self.conv3,
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        #     # --------------------------  4 conv-слой ---------------------------
        #     self.conv4,
        # )
        # self.seq_fc = nn.Sequential(
        #     # --------------------------  fc-слои ---------------------------
        #     nn.Linear(64 * 26, 2048),
        #     nn.ReLU(),
        #     nn.LayerNorm(2048),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.LayerNorm(1024),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.LayerNorm(512),
        #     nn.Linear(512, out_channels),  # N - количество ненулевых элементов матрицы связи
        #     nn.Tanh()
        # )
        # --------------------------  1 conv-слой ---------------------------
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding='same')
        # --------------------------  2 conv-слой ---------------------------
        self.conv2 = nn.Sequential(
                                   nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same'))
        # --------------------------  3 conv-слой ---------------------------
        self.conv3 = nn.Sequential(
                                   nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same'))
        # --------------------------  4 conv-слой ---------------------------
        self.conv4 = nn.Sequential(
                                   nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same'))
        self.seq_conv = torch.nn.Sequential(
            # --------------------------  1 conv-слой ---------------------------
            self.conv1,
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            # --------------------------  2 conv-слой ---------------------------
            self.conv2,
            nn.LeakyReLU(),
            self.conv2,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # --------------------------  3 conv-слой ---------------------------
            self.conv3,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # --------------------------  4 conv-слой ---------------------------
            self.conv4,
        )
        self.seq_fc = nn.Sequential(
            # --------------------------  fc-слои ---------------------------
            nn.Linear(26*64, 512),
            nn.Softsign(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.Softsign(),
            nn.LayerNorm(512),
            nn.Linear(512, out_channels), # N - количество ненулевых элементов матрицы связи
            nn.Softsign()
        )

    def encode(self, x):
        conv_x = self.seq_conv(x)
        conv_x_reshaped = conv_x.view(conv_x.size(0), -1)
        fc_x = self.seq_fc(conv_x_reshaped)
        return fc_x

    def forward(self, x):
        encoded = self.encode(x)
        return encoded


class ResNet1DBiRNN(nn.Module):
    def __init__(self, in_channels=8, resnet_out_channels=512,
                 hidden_size=64, num_layers=3, out_channels=10,
                 dropout=0.2, rnn_type='lstm'):
        super().__init__()

        # ResNet1D часть
        self.resnet = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ResNet1D.make_layer(64, 64, 2, stride=1),
            ResNet1D.make_layer(64, 128, 2, stride=2),
            ResNet1D.make_layer(128, resnet_out_channels, 2, stride=2),
            # ResNet1D.make_layer(256, resnet_out_channels, 2, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )

        # BiRNN часть
        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU if rnn_type == 'gru' else nn.RNN
        self.rnn = rnn_class(
            input_size=resnet_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size * 2, out_channels)
        self.hidden_size = hidden_size

    def forward(self, x):
        # Применяем ResNet1D
        x = self.resnet(x)  # [batch, resnet_out_channels, 1]

        # Подготовка для RNN: [batch, seq_len=1, features]
        x = x.permute(0, 2, 1)  # [batch, 1, resnet_out_channels]

        # Применяем BiRNN
        out, _ = self.rnn(x)

        # Объединяем направления
        out = torch.cat([out[:, -1, :self.hidden_size],
                         out[:, 0, self.hidden_size:]], dim=1)

        # Финальный классификатор
        return self.fc(out)


class BiRNNResNet1D(nn.Module):
    def __init__(self, in_channels=8, rnn_hidden_size=64, rnn_layers=3,
                 resnet_in_channels=128, resnet_out_channels=512, out_channels=10,
                 rnn_dropout=0.2, rnn_type='lstm'):
        super().__init__()

        # BiRNN часть
        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU if rnn_type == 'gru' else nn.RNN
        self.birnn = rnn_class(
            input_size=in_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0
        )

        # Преобразование выхода BiRNN для ResNet1D
        self.rnn_to_resnet = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, resnet_in_channels),
            nn.ReLU()
        )

        # ResNet1D часть
        self.resnet = nn.Sequential(
            nn.Conv1d(resnet_in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ResNet1D.make_layer(64, 64, 2, stride=1),
            ResNet1D.make_layer(64, 128, 2, stride=2),
            ResNet1D.make_layer(128, 256, 2, stride=2),
            ResNet1D.make_layer(256, resnet_out_channels, 2, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(resnet_out_channels, out_channels)

    def forward(self, x):
        # BiRNN обработка
        rnn_out, _ = self.birnn(x)  # [batch, seq_len, hidden_size*2]

        # Преобразование для ResNet1D
        features = self.rnn_to_resnet(rnn_out)  # [batch, seq_len, resnet_in_channels]

        # Подготовка для Conv1d (добавляем dimension канала)
        features = features.unsqueeze(1)  # [batch, 1, seq_len, resnet_in_channels]
        features = features.permute(0, 1, 3, 2)  # [batch, 1, resnet_in_channels, seq_len]
        features = features.squeeze(1)  # [batch, resnet_in_channels, seq_len]

        # ResNet1D обработка
        resnet_out = self.resnet(features)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)

        return self.fc(resnet_out)


class ResNetRNN(nn.Module):
    def __init__(self,
                 resnet_model,  # Экземпляр ResNet1DFlexible
                 rnn_type='LSTM',  # 'LSTM' или 'GRU'
                 rnn_hidden_size=128,
                 rnn_num_layers=1,
                 rnn_bidirectional=False,
                 out_features=1,  # Размер выходного вектора
                 dropout=0.0):
        super().__init__()

        self.resnet = resnet_model

        # Получаем размер выхода ResNet (число каналов после avgpool + fc)
        dummy_input = torch.randn(1, resnet_model.conv1.in_channels, 64)
        with torch.no_grad():
            resnet_out = resnet_model(dummy_input)
        resnet_out_dim = resnet_out.shape[-1]

        # RNN: LSTM или GRU
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(input_size=resnet_out_dim,
                           hidden_size=rnn_hidden_size,
                           num_layers=rnn_num_layers,
                           bidirectional=rnn_bidirectional,
                           batch_first=True,
                           dropout=dropout if rnn_num_layers > 1 else 0)

        # Выходной слой
        rnn_out_dim = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.fc_out = nn.Linear(rnn_out_dim, out_features)

    def forward(self, x):
        """
        x: (batch_size, channels, points_num)
        """
        # Пропускаем через ResNet
        batch_size = x.size(0)

        # Получаем признаковое представление
        resnet_features = self.resnet(x)
        # resnet_features: (batch_size, resnet_out_dim)

        # Для RNN требуется размерность (batch, seq_len, feature)
        # Если seq_len=1 (один шаг), можем expand или reshape
        resnet_features = resnet_features.unsqueeze(1)  # (batch_size, 1, feature)

        # Пропускаем через RNN
        rnn_out, _ = self.rnn(resnet_features)  # (batch_size, 1, hidden)
        rnn_out = rnn_out.squeeze(1)  # (batch_size, hidden)

        # Финальный выход
        out = self.fc_out(rnn_out)
        return out


class MixerBlock1D(nn.Module):
    def __init__(self, num_tokens, num_channels, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.token_norm = nn.LayerNorm(num_channels)  # по каналам
        self.token_fc1 = nn.Linear(num_tokens, token_mlp_dim)
        self.token_act = nn.GELU()
        self.token_fc2 = nn.Linear(token_mlp_dim, num_tokens)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_fc1 = nn.Linear(num_channels, channel_mlp_dim)
        self.channel_act = nn.GELU()
        self.channel_fc2 = nn.Linear(channel_mlp_dim, num_channels)

    def forward(self, x):
        # x: (B, T, C)
        y = self.token_norm(x)  # LayerNorm по каналам (последняя ось C)

        # Для token mixing хотим применить линейный слой по токенам (T)
        # Линейный слой ожидает признаки на последней оси,
        # поэтому меняем местами T и C:
        y = y.transpose(1, 2)  # (B, C, T)

        y = self.token_fc1(y)
        y = self.token_act(y)
        y = self.token_fc2(y)

        y = y.transpose(1, 2)  # обратно в (B, T, C)
        x = x + y

        # channel mixing:
        z = self.channel_norm(x)
        z = self.channel_fc1(z)
        z = self.channel_act(z)
        z = self.channel_fc2(z)

        return x + z


class MLPMixer1D(nn.Module):
    def __init__(self,
                 num_tokens=301,
                 num_channels=8,
                 token_mlp_dim=256,
                 channel_mlp_dim=512,
                 num_blocks=4,
                 out_dim=30):
        super().__init__()
        self.mixer_blocks = nn.Sequential(*[
            MixerBlock1D(num_tokens, num_channels, token_mlp_dim, channel_mlp_dim)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(num_channels)
        self.head = nn.Linear(num_channels, out_dim)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        x = self.mixer_blocks(x)
        x = self.norm(x)  # нормируем по каналам
        x = x.mean(dim=1)  # усреднение по токенам
        return self.head(x)

class ResNetRNNSequence(nn.Module):
    def __init__(self,
                 resnet_model,  # Экземпляр ResNet1DFlexible (без avgpool!)
                 rnn_type='LSTM',  # 'LSTM' или 'GRU'
                 rnn_hidden_size=128,
                 rnn_num_layers=1,
                 rnn_bidirectional=False,
                 out_features=1,  # Выходной размер
                 dropout=0.0,
                 return_sequences=False):  # True: выход всей последовательности
        super().__init__()

        self.resnet = resnet_model
        self.return_sequences = return_sequences

        # Удаляем avgpool и fc ResNet-а, т.к. они делают "глобальное усреднение"
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        # Примерный размер выхода после ResNet
        dummy_input = torch.randn(1, resnet_model.conv1.in_channels, 128)
        with torch.no_grad():
            resnet_features = self.resnet(dummy_input)
            feature_dim = resnet_features.shape[1]
            seq_len = resnet_features.shape[2]

        # RNN
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(input_size=feature_dim,
                           hidden_size=rnn_hidden_size,
                           num_layers=rnn_num_layers,
                           bidirectional=rnn_bidirectional,
                           batch_first=True,
                           dropout=dropout if rnn_num_layers > 1 else 0)

        # Финальный FC слой
        rnn_out_dim = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.fc_out = CorrectionMLP(input_dim=rnn_out_dim, output_dim=out_features,
                                    hidden_dims=[128, 256, 512])
        # self.fc_out = nn.Linear(rnn_out_dim, out_features)

    def forward(self, x):
        """
        x: (batch_size, channels, points_num)
        """
        # Получаем последовательность признаков
        features = self.resnet(x)  # (batch, feature_dim, seq_len)

        # Транспонируем в (batch, seq_len, feature_dim)
        features = features.permute(0, 2, 1)

        # Пропускаем через RNN
        rnn_out, _ = self.rnn(features)  # (batch, seq_len, hidden)

        if self.return_sequences:
            # Выдаем все временные шаги
            out = self.fc_out(rnn_out)  # (batch, seq_len, out_features)
        else:
            # Берем только последний временной шаг
            last_out = rnn_out[:, -1, :]  # (batch, hidden)
            out = self.fc_out(last_out)  # (batch, out_features)

        return out


class MultiChannel1Dto2D(nn.Module):
    def __init__(self, input_channels=8, output_size=(64, 80), latent_channels=64,
                 selected_indices=None):
        super().__init__()
        self.output_size = output_size
        self.latent_H, self.latent_W = output_size  # латентный размер (для reshape перед декодером)
        self.latent_channels = latent_channels

        self.selected_indices_rc = selected_indices
        if selected_indices is not None:
            # Преобразуем список (row, col) в плоские индексы [row * W + col]
            H, W = output_size
            flat_indices = [r * W + c for r, c in selected_indices]
            self.selected_indices = torch.tensor(flat_indices, dtype=torch.long)
        else:
            self.selected_indices = None

        self.encoder = ResNet1DFlexible(
            in_channels=input_channels,
            out_channels=self.latent_channels * self.latent_H * self.latent_W,
            num_blocks=[1, 4, 3, 5],
            layer_channels=[64, 64, 128, 256],
            first_conv_kernel=8,
            first_conv_channels=64,
            activation_in='sigmoid',
            activation_block='swish',
            use_se=False,
            se_reduction=1
        )

        # self.encoder = nn.Sequential(
        #     nn.BatchNorm1d(input_channels),
        #     nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(32),
        #     nn.Conv1d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten(),
        #     nn.LayerNorm(64),
        #     nn.Linear(64, self.latent_channels * self.latent_H * self.latent_W),
        #     nn.ReLU()
        # )

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(self.latent_channels),
            nn.ConvTranspose2d(self.latent_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.encoder(x)
        x = x.view(B, self.latent_channels, self.latent_H, self.latent_W)
        x = self.decoder(x)  # [B, 1, H, W]
        x = x.squeeze(1)  # [B, H, W]

        if self.selected_indices is not None:
            # Вынимаем выбранные элементы
            x_flat = x.view(B, -1)  # [B, H*W]
            out = x_flat[:, self.selected_indices.to(x.device)]
            return out

        return x
        if self.selected_indices is not None:
            # selected_indices: [num_indices]
            # нужно извлечь эти элементы из каждого примера батча
            # reshape в [B, H*W]
            x_flat = x.view(B, -1)
            # используем индексы для отбора по последнему измерению
            out = x_flat[:, self.selected_indices.to(x.device)]  # [B, num_indices]
            return out

        return x
class Swish(nn.Module):
    """Активация Swish (Silu)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class MBConv1DBlock(nn.Module):
    """MBConv-блок для 1D данных (аналог MobileNetV2 + SE-блок)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        # Expansion phase (если expand_ratio != 1)
        self.expand = None
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv1d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm1d(expanded_channels),
                Swish(),
            )

        # Depthwise-свертка
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                expanded_channels,
                expanded_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm1d(expanded_channels),
            Swish(),
        )

        # Squeeze-and-Excitation (SE-блок)
        squeezed_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(expanded_channels, squeezed_channels, 1),
            Swish(),
            nn.Conv1d(squeezed_channels, expanded_channels, 1),
            nn.Sigmoid(),
        )

        # Pointwise-свертка
        self.project = nn.Sequential(
            nn.Conv1d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        if self.expand:
            x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x) * x  # SE-блок
        x = self.project(x)
        if self.use_residual:
            x = self.dropout(x)
            x = x + residual
        return x



class EfficientNet1D(nn.Module):
    """1D-версия EfficientNet с настраиваемыми гиперпараметрами"""
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,  # Для регрессии
        width_coeff: float = 1.0,
        depth_coeff: float = 1.0,
        dropout_rate: float = 0.2,
        se_ratio: float = 0.25,
        stochastic_depth: bool = False,
    ):
        super().__init__()
        # Базовые параметры блоков (аналоги EfficientNet-B0)
        base_channels = [32, 16, 24, 40, 80, 112, 192, 320]
        base_depths = [1, 2, 2, 3, 3, 4, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        expand_ratios = [1, 6, 6, 6, 6, 6, 6]

        # Масштабирование ширины и глубины
        channels = [math.ceil(c * width_coeff) for c in base_channels]
        depths = [math.ceil(d * depth_coeff) for d in base_depths]

        # Первая свертка
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            Swish(),
        )

        # MBConv-блоки
        blocks = []
        for i in range(7):
            for j in range(depths[i]):
                stride = strides[i] if j == 0 else 1
                blocks.append(
                    MBConv1DBlock(
                        in_channels=channels[i] if j == 0 else channels[i + 1],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_sizes[i],
                        stride=stride,
                        expand_ratio=expand_ratios[i],
                        se_ratio=se_ratio,
                        dropout_rate=dropout_rate,
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        # Финальные слои
        self.head = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], 1, bias=False),
            nn.BatchNorm1d(channels[-1]),
            Swish(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x



# ────────────────────────────────────────────────────────────────────────────────
#   BasicBlock2D (аналог обычного блока ResNet, но в 2‑D)
# ────────────────────────────────────────────────────────────────────────────────
class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = 1,
        activation: str = "relu",
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # сокращённый SE‑блок (по желанию)
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // se_reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // se_reduction, out_channels, kernel_size=1),
                nn.Sigmoid(),
            )

        # если размерность или stride изменились — делаем shortcut‑проекцию
        self.shortcut = (
            nn.Identity()
            if (in_channels == out_channels and stride == 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_se:
            scale = self.se(out)
            out = out * scale

        out += identity
        out = self.act(out)
        return out


# ────────────────────────────────────────────────────────────────────────────────
#   ResNet‑Flexible 2‑D
# ────────────────────────────────────────────────────────────────────────────────
class ResNet2DFlexible(nn.Module):
    """
    Вход (без частот):  (B, C_in, N)
    После добавления f: (B, 1, C_in+1, N)  → Conv2d
    Выход: (B, out_channels)
    """

    def __init__(
        self,
        freq_vector: torch.Tensor,          # ← 1‑D тензор нормированных частот
        in_channels: int = 8,               # число исходных S‑каналов
        out_channels: int = 30,             # размер выходного вектора
        first_conv_channels: int = 64,
        first_conv_kernel: int = 7,
        first_maxpool_kernel: int = 3,
        layer_channels: list[int] = [64, 128, 256, 512],
        # layer_channels: list[int] = [128, 256, 256],
        # num_blocks: list[int] = [2, 1, 3],
        num_blocks: list[int] = [2, 2, 2, 2],
        activation_in: str = "relu",
        activation_block: str = "relu",
        use_se: bool = True,
        se_reduction: int = 16,
    ):
        super().__init__()

        self.activation = get_activation(activation_in)

        # ----------  Сохраняем / регистрируем вектор частот  -------------------
        # freq_vector должен иметь shape (N,)
        self.register_buffer("freq_vector", freq_vector.float().unsqueeze(0))  # (1, N)

        # ----------  Первый 2‑D conv  -----------------------------------------
        #   in_channels_2d = 1 (мы подадим 1 "карту", где высота = C_in+1)
        self.conv1 = nn.Conv2d(
            1,
            first_conv_channels,
            kernel_size=(3, first_conv_kernel),
            stride=2,
            padding=first_conv_kernel // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(first_conv_channels)
        self.maxpool = nn.MaxPool2d(
            kernel_size=first_maxpool_kernel,
            stride=2,
            padding=first_maxpool_kernel // 2,
        )

        # ----------  ResNet‑stack  --------------------------------------------
        self.layers = nn.ModuleList()
        in_ch = first_conv_channels
        for out_ch, n_blocks in zip(layer_channels, num_blocks):
            stride = 2 if in_ch != out_ch else 1
            self.layers.append(
                self.make_layer(
                    in_ch,
                    out_ch,
                    n_blocks,
                    stride=stride,
                    activation=activation_block,
                    use_se=use_se,
                    se_reduction=se_reduction,
                )
            )
            in_ch = out_ch

        # ----------  Голова  ---------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, out_channels)

        # ----------  Shortcut‑ветвь для исходного 1‑D сигнала ------------------
        self.shortcut = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AdaptiveAvgPool1d(1),
        )

    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def make_layer(in_channels, out_channels, num_blocks, stride,
                   activation, use_se, se_reduction):
        layers = [
            BasicBlock2D(in_channels, out_channels, stride,
                         activation, use_se, se_reduction)
        ]
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock2D(out_channels, out_channels, (1, 1),
                             activation, use_se, se_reduction)
            )
        return nn.Sequential(*layers)

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, x):
        """
        x: (B, C_in, N)
        """
        B, C, N = x.shape

        # -- 1. shortcut‑ветка (1‑D) для резидуального сложения на выходе
        identity = self.shortcut(x)          # (B, out_channels, 1)
        identity = identity.view(B, -1)      # (B, out_channels)

        # -- 2. добавляем строку частот
        #    freq_vector : (1, N) → (B, 1, N)
        f = self.freq_vector.repeat(B, 1)        # (B, N)
        f = f.unsqueeze(1)                       # (B, 1, N)
        x = torch.cat([x, f], dim=1)             # (B, C_in+1, N)

        # -- 3. превращаем в 2‑D: добавляем channel‑axis =1 → (B,1,H,W)
        x = x.unsqueeze(1)                       # (B, 1, H=C+1, W=N)

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)      # (B, in_ch, 1, 1)
        x = x.view(B, -1)        # (B, in_ch)
        x = self.fc(x)           # (B, out_channels)

        # -- 4. добавляем shortcut‑ветку
        x += identity
        return x


import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from lightning.pytorch.callbacks import Callback

# --- Base Analyzer ---
class BlockImportanceAnalyzer:
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = copy.deepcopy(model).to(self.device)
        # collect blocks as (layer_idx, block_idx)
        # if type(model) == ModelWithCorrection:
        #     self.blocks = [(li, bi)
        #                    for li, layer in enumerate(self.model.main.layers)
        #                    for bi, _ in enumerate(layer)]
        # else:
        self.blocks = [(li, bi)
                       for li, layer in enumerate(self.model.model._main_model.layers)
                       for bi, _ in enumerate(layer)]

    def compute_r2(self, dataloader):
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                preds.append(out.cpu().numpy())
                targets.append(y.cpu().numpy())
        return r2_score(np.concatenate(targets), np.concatenate(preds))

    def get_block(self, model, li, bi):
        if type(model) == filters.mwfilter_lightning.MWFilterBaseLMWithMetrics:
            return model.model._main_model.layers[li][bi]
        else:
            return model._main_model.layers[li][bi]

# --- Callbacks ---
class AblationCallback(Callback):
    """Ablation: drop in R2 when bypassing each block"""
    def __init__(self, val_dataloader):
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        analyzer = BlockImportanceAnalyzer(pl_module)
        baseline = analyzer.compute_r2(self.val_dataloader)
        drops = {}
        orig = analyzer.model
        for li, bi in analyzer.blocks:
            m_copy = copy.deepcopy(orig)
            blk = analyzer.get_block(m_copy, li, bi)
            # bypass via shortcut
            blk.forward = lambda x, sc=blk.shortcut: sc(x)
            r2 = BlockImportanceAnalyzer(m_copy).compute_r2(self.val_dataloader)
            drops[f"{li}-{bi}"] = baseline - r2
        # plot
        labels, values = zip(*drops.items())
        plt.figure()
        plt.bar(labels, values)
        plt.xticks(rotation=90)
        plt.title(f"Ablation importance (drop R2), baseline={baseline:.3f}")
        plt.tight_layout()
        # plt.show()

class WeightNormCallback(Callback):
    """Compute L1-norm of weights for each block"""
    def on_validation_epoch_end(self, trainer, pl_module):
        analyzer = BlockImportanceAnalyzer(pl_module)
        norms = {}
        for li, bi in analyzer.blocks:
            blk = analyzer.get_block(analyzer.model, li, bi)
            norms[f"{li}-{bi}"] = sum(p.abs().sum().item() for p in blk.parameters())
        labels, values = zip(*norms.items())
        plt.figure()
        plt.bar(labels, values)
        plt.xticks(rotation=90)
        plt.title("Weight L1-norm per block")
        plt.tight_layout()
        # plt.show()

class GradNormCallback(Callback):
    """Compute gradient norm per block on one batch"""
    def __init__(self, sample_batch):
        self.sample_batch = sample_batch

    def on_validation_epoch_end(self, trainer, pl_module):
        analyzer = BlockImportanceAnalyzer(pl_module)
        x, y = self.sample_batch
        x, y = x.to(pl_module.device), y.to(pl_module.device)
        # enable grad for validation
        with torch.enable_grad():
            pl_module.model.train()
            pl_module.model.zero_grad()
            out = pl_module.model(x)
            loss = nn.MSELoss()(out, y)
            loss.backward()
        grads = {}
        for li, bi in analyzer.blocks:
            blk = analyzer.get_block(pl_module.model, li, bi)
            grads[f"{li}-{bi}"] = sum(
                p.grad.norm().item() for p in blk.parameters() if p.grad is not None
            )
        # reset model state
        pl_module.model.zero_grad()
        pl_module.model.eval()
        # plot
        labels, values = zip(*grads.items())
        plt.figure()
        plt.bar(labels, values)
        plt.xticks(rotation=90)
        plt.title("Gradient norm per block")
        plt.tight_layout()
        # plt.show()

class ActivationCallback(Callback):
    """Compute mean activation per block on one batch"""
    def __init__(self, sample_batch):
        self.sample_batch = sample_batch

    def on_validation_epoch_end(self, trainer, pl_module):
        analyzer = BlockImportanceAnalyzer(pl_module)
        x, _ = self.sample_batch
        x = x.to(pl_module.device)
        acts = {}
        handles = []
        for li, bi in analyzer.blocks:
            blk = analyzer.get_block(pl_module.model, li, bi)
            handles.append(blk.register_forward_hook(
                lambda m, i, o, key=f"{li}-{bi}": acts.setdefault(key, o.abs().mean().item())
            ))
        with torch.no_grad():
            pl_module.model(x)
        for h in handles: h.remove()
        labels, values = zip(*acts.items())
        plt.figure()
        plt.bar(labels, values)
        plt.xticks(rotation=90)
        plt.title("Mean activation per block")
        plt.tight_layout()
        # plt.show()

class TorchvisionModels(nn.Module):
    def __init__(self,
                 freq_vector: torch.Tensor,  # ← 1‑D тензор нормированных частот
                 model: nn.Module
                 ):
        super().__init__()
        self.register_buffer("freq_vector", freq_vector.float())  # (N)
        self.model = model

    def forward(self, x):
        x = self.expand_to_2d(x, self.freq_vector)
        x = self.model(x)
        return x

    import torch

    def expand_to_2d(self, x_1d: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # x_2d = torch.einsum('bcn,f->bcnf', x_1d, freqs)
        """
        Преобразует тензор (B, 8, 301) + вектор частот (301,) → (B, 1, 9, 301)
        """
        B, C, N = x_1d.shape  # B=64, C=8, N=301
        freq_vector = self.freq_vector.unsqueeze(0)
        f = freq_vector.repeat(B, 1)        # (B, N)
        f = f.unsqueeze(1)                       # (B, 1, N)
        x_2d = torch.cat([x_1d, f], dim=1)
        x_2d = x_2d.unsqueeze(1)  # (B, 1, H=C+1, W=N)
        """
        # Преобразует тензор (B, 8, 301) + вектор частот (301,) → (B, 8, 301, 301)
        # """
        # assert freqs.shape[0] == N, "Число отсчётов должно совпадать"
        #
        # # x_1d: (B, 8, 301) → (B, 8, 301, 1), потом expand до (B, 8, 301, 301)
        # x_expanded = x_1d.unsqueeze(-1).expand(-1, -1, -1, N)
        #
        # # freqs: (301,) → (1, 1, 1, 301), потом expand до (B, 8, 301, 301)
        # freqs_expanded = freqs.view(1, 1, 1, N).expand(B, C, N, N)
        #
        # # Объединяем данные: можно сложить, умножить или склеить как отдельный канал
        # # Пример: просто добавим частоты к данным
        # x_2d = x_expanded + freqs_expanded

        return x_2d


# --- Фиксированный depthwise FIR: сглаживает высокочастотную рябь после апсемплинга ---
class AntiAlias1D(nn.Module):
    """
    Простой низкочастотный depthwise FIR (по каналам), taps=5 (биномиальный [1,4,6,4,1]/16).
    Не обучаемый, длину не меняет (padding=taps//2).
    """
    def __init__(self, channels: int, taps: int = 5):
        super().__init__()
        assert taps in (3, 5, 7), "Рекомендуемые taps: 3,5,7"
        if taps == 3:
            k = torch.tensor([1., 2., 1.])
        elif taps == 5:
            k = torch.tensor([1., 4., 6., 4., 1.])
        else:  # 7
            k = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
        k = (k / k.sum()).view(1, 1, -1)                   # (1,1,K)

        self.fir = nn.Conv1d(
            channels, channels, kernel_size=taps,
            padding=taps // 2, groups=channels, bias=False
        )
        # with torch.no_grad():
        #     self.fir.weight.copy_(k.repeat(channels, 1, 1))  # (C,1,K)
        # for p in self.fir.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        return self.fir(x)


# --- Обёртка над ConvTranspose1d: deconv -> AntiAlias -> активация ---
class DeconvAA1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding=0,
                 taps=5, act=nn.SiLU):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
        self.aa = AntiAlias1D(out_ch, taps=taps)
        self.act = act()

    def forward(self, x):
        x = self.deconv(x)
        x = self.aa(x)
        x = self.act(x)
        return x


# --- Свёртка с reflect-паддингом (лучше краев, чем zero-pad) ---
class ReflectConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.stride = stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=0, bias=bias)
    def forward(self, x):
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode='reflect')
        return self.conv(x)

# --- Блок апсемплинга: интерполяция + Conv1d(+reflect) + SiLU ---
class UpBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = ReflectConv1d(in_ch, out_ch, kernel_size=k, stride=1, bias=True)
        self.act  = nn.SiLU()
    def forward(self, x, target_len):
        # Либо scale_factor=2, либо точный target_len — удобнее второе, чтобы попасть в 151/301
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        x = self.conv(x)
        return self.act(x)


class ConvAE(nn.Module):
    # def __init__(self, in_ch=8, z_dim=32):
    #     super().__init__()
    #     # -------- Encoder --------
    #     self.enc = nn.Sequential(
    #         nn.Conv1d(in_ch, 32, kernel_size=5, stride=2, padding=2),  # 301 -> 151
    #         nn.ReLU(inplace=True),
    #         nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 151 -> 76
    #         nn.ReLU(inplace=True),
    #         nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # 76 -> 76
    #         nn.ReLU(inplace=True),
    #     )
    #     self.to_z = nn.Linear(128 * 76, z_dim)
    #
    #     # -------- Decoder --------
    #     self.from_z = nn.Linear(z_dim, 128 * 76)
    #     self.dec = nn.Sequential(
    #         nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),  # 76 -> 76
    #         nn.ReLU(inplace=True),
    #         nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2),  # 76 -> 151
    #         nn.ReLU(inplace=True),
    #         nn.ConvTranspose1d(32, in_ch, kernel_size=5, stride=2, padding=2),  # 151 -> 301
    #         nn.Sigmoid(),  # если вход нормирован [0,1]; иначе замените
    #     )
    #
    # def encode(self, x):  # x: (B, 8, 301)
    #     h = self.enc(x)  # (B, 128, 38)
    #     z = self.to_z(h.view(x.size(0), -1))  # (B, z_dim)
    #     return z
    #
    # def decode(self, z):  # z: (B, z_dim)
    #     h = self.from_z(z).view(z.size(0), 128, 76)
    #     x_hat = self.dec(h)  # (B, 8, 301)
    #     return x_hat
    #
    # def forward(self, x):
    #     z = self.encode(x)
    #     x_hat = self.decode(z)
    #     return x_hat, z
    def __init__(self, in_ch=8, z_dim=32, aa_taps=5):
        super().__init__()
        # -------- Encoder --------
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, stride=2, padding=2),  # 301 -> 151
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 151 -> 76
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # 76 -> 76
            nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # 76 -> 38  (новый слой)
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),  # 38 -> 38  (новый слой)
            nn.SiLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),  # 38 -> 19  (новый слой)
            nn.SiLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),  # 19 -> 19  (новый слой)
            nn.SiLU(),
        )
        # стало 128 * 38 признаков
        self.to_z = nn.Sequential(
            nn.Linear(512 * 19, z_dim),
            # nn.Linear(512 * 19, 256),
            # nn.SiLU(),
            # nn.Linear(256, z_dim),
        )

        # -------- Decoder --------
        self.from_z = nn.Sequential(
            nn.Linear(z_dim, 512 * 19),
            # nn.Linear(z_dim, 256),
            # nn.SiLU(),
            # nn.Linear(256, 512 * 19),
        )
        self.dec = nn.Sequential(
            # 38 -> 76 (нужен output_padding=1 для точной длины)
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),  # 19 -> 38  (новый слой)
            nn.SiLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            # 38 -> 76 (нужен output_padding=1 для точной длины)
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1),  # 38 -> 38  (новый слой)
            nn.SiLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            # 76 -> 76
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            # 76 -> 151
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            # 151 -> 301
            nn.ConvTranspose1d(32, in_ch, kernel_size=5, stride=2, padding=2),
            nn.Sigmoid()
        )

    def encode(self, x):  # x: (B, 8, 301)
        h = self.enc(x)  # (B, 128, 38)
        z = self.to_z(h.view(x.size(0), -1))  # (B, z_dim)
        return z

    def decode(self, z):  # z: (B, z_dim)
        h = self.from_z(z).view(z.size(0), 512, 19)  # (B, 128, 38)
        x_hat = self.dec(h)  # (B, 8, 301)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    # def __init__(self, in_ch=8, z_dim=32, out_act='sigmoid'):
    #     super().__init__()
    #
    #     # ---------- Encoder (301 -> 19) ----------
    #     self.enc = nn.Sequential(
    #         ReflectConv1d(in_ch, 32, kernel_size=5, stride=2),  # 301 -> 151
    #         nn.SiLU(),
    #         ReflectConv1d(32, 64, kernel_size=5, stride=2),  # 151 -> 76
    #         nn.SiLU(),
    #         ReflectConv1d(64, 128, kernel_size=3, stride=1),  # 76  -> 76
    #         nn.SiLU(),
    #         ReflectConv1d(128, 256, kernel_size=3, stride=2),  # 76  -> 38
    #         nn.SiLU(),
    #         ReflectConv1d(256, 256, kernel_size=3, stride=1),  # 38  -> 38
    #         nn.SiLU(),
    #         ReflectConv1d(256, 512, kernel_size=3, stride=2),  # 38  -> 19
    #         nn.SiLU(),
    #         ReflectConv1d(512, 512, kernel_size=3, stride=1),  # 19  -> 19
    #         nn.SiLU(),
    #     )
    #
    #     self.to_z = nn.Linear(512 * 19, z_dim)
    #     self.from_z = nn.Linear(z_dim, 512 * 19)
    #
    #     # ---------- Decoder: только Upsample + Conv1d ----------
    #     self.pre = nn.Sequential(ReflectConv1d(512, 512, kernel_size=3, stride=1), nn.SiLU())  # 19 -> 19
    #     self.up1 = UpBlock1D(512, 256, k=3)  # 19  -> 38
    #     self.ref1 = nn.Sequential(ReflectConv1d(256, 256, kernel_size=3, stride=1), nn.SiLU())
    #
    #     self.up2 = UpBlock1D(256, 128, k=3)  # 38  -> 76
    #     self.ref2 = nn.Sequential(ReflectConv1d(128, 128, kernel_size=3, stride=1), nn.SiLU())
    #
    #     self.up3 = UpBlock1D(128, 64, k=5)  # 76  -> 151
    #     self.ref3 = nn.Sequential(ReflectConv1d(64, 64, kernel_size=3, stride=1), nn.SiLU())
    #
    #     self.up4 = UpBlock1D(64, 32, k=5)  # 151 -> 301
    #     self.ref4 = nn.Sequential(ReflectConv1d(32, 32, kernel_size=3, stride=1), nn.SiLU())
    #
    #     self.out = ReflectConv1d(32, in_ch, kernel_size=5, stride=1)
    #
    #     if out_act == 'sigmoid':
    #         self.out_act = nn.Sigmoid()
    #     elif out_act == 'tanh':
    #         self.out_act = nn.Tanh()
    #     else:
    #         self.out_act = nn.Identity()
    #
    #     # ---------- API ----------
    #
    # def encode(self, x):  # x: (B, 8, 301)
    #     # Сохраним длины на ключевых уровнях, чтобы точно восстановиться
    #     e1 = self.enc[0:2](x)  # 151
    #     e2 = self.enc[2:4](e1)  # 76
    #     e3 = self.enc[4:6](e2)  # 76
    #     e4 = self.enc[6:8](e3)  # 38
    #     e5 = self.enc[8:10](e4)  # 38
    #     e6 = self.enc[10:12](e5)  # 19
    #     e7 = self.enc[12:](e6)  # 19
    #
    #     h = e7
    #     z = self.to_z(h.flatten(1))
    #     # Вернём нужные целевые длины (берём из фактических тензоров)
    #     lens = {
    #         "L1": x.size(-1),  # 301
    #         "L2": e1.size(-1),  # 151
    #         "L3": e2.size(-1),  # 76
    #         "L4": e4.size(-1),  # 38
    #         "L5": e7.size(-1),  # 19 (бутылочное)
    #     }
    #     return z, lens
    #
    # def decode(self, z, lens):
    #     B = z.size(0)
    #     h = self.from_z(z).view(B, 512, lens["L5"])  # 19
    #     h = self.pre(h)  # 19
    #
    #     h = self.up1(h, target_len=lens["L4"])  # 19 -> 38
    #     h = self.ref1(h)  # 38
    #
    #     h = self.up2(h, target_len=lens["L3"])  # 38 -> 76
    #     h = self.ref2(h)  # 76
    #
    #     h = self.up3(h, target_len=lens["L2"])  # 76 -> 151
    #     h = self.ref3(h)  # 151
    #
    #     h = self.up4(h, target_len=lens["L1"])  # 151 -> 301
    #     h = self.ref4(h)  # 301
    #
    #     y = self.out(h)  # 301
    #     return self.out_act(y)
    #
    # def forward(self, x):
    #     z, lens = self.encode(x)
    #     x_hat = self.decode(z, lens)
    #     return x_hat, z

    # def __init__(self, in_ch=8, z_dim=32):
    #     super().__init__()
    #     # -------- Encoder --------
    #     # 301 -> 152 -> 76 -> 76  (после паддинга до 304 будет 304->152->76)
    #     self.enc = nn.Sequential(
    #         nn.Conv1d(in_ch, 32, kernel_size=5, stride=2, padding=2),  # ~L/2
    #         nn.BatchNorm1d(32),
    #         nn.Tanh(),
    #         nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),     # ~L/4
    #         nn.BatchNorm1d(64),
    #         nn.Tanh(),
    #         nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),    # длина та же
    #         nn.BatchNorm1d(128),
    #         nn.Tanh(),
    #     )
    #     # после двух stride=2 получим L_enc = ceil(L/4). Для L=304 это 76.
    #     self.to_z = nn.Linear(128 * 76, z_dim)
    #
    #     # -------- Decoder --------
    #     self.from_z = nn.Linear(z_dim, 128 * 76)
    #     self.dec = nn.Sequential(
    #         nn.BatchNorm1d(128),
    #         nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=0), # 76 -> 152
    #         nn.Tanh(),
    #         nn.BatchNorm1d(64),
    #         nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0),  # 152 -> 304
    #         nn.Tanh(),
    #         nn.BatchNorm1d(32),
    #         nn.Conv1d(32, in_ch, kernel_size=3, stride=1, padding=1),  # 304 -> 304 (чистка артефактов)
    #         nn.Sigmoid(),  # если вход нормирован в [0,1]
    #     )
    #
    # @staticmethod
    # def _pad_to_multiple_of_4(x):
    #     # x: (B,C,L). Паддим справа нулями до ближайшей длины, кратной 4
    #     L = x.size(-1)
    #     need = (4 - (L % 4)) % 4
    #     if need == 0:
    #         return x, 0
    #     return F.pad(x, (0, need)), need
    #
    # def encode(self, x):
    #     x_pad, pad = self._pad_to_multiple_of_4(x)          # -> L=304
    #     h = self.enc(x_pad)                                  # (B,128,76)
    #     z = self.to_z(h.flatten(1))                          # (B,z_dim)
    #     return z, pad
    #
    # def decode(self, z, pad):
    #     h = self.from_z(z).view(z.size(0), 128, 76)          # (B,128,76)
    #     x_hat_pad = self.dec(h)                              # (B,8,304)
    #     # срезаем паддинг обратно к исходной длине 301
    #     L_target = 301
    #     x_hat = x_hat_pad[..., :L_target]
    #     return x_hat
    #
    # def forward(self, x):
    #     z, pad = self.encode(x)          # pad храню на случай, если захочешь другой L
    #     x_hat = self.decode(z, pad)
    #     return x_hat, z



class RNNAutoencoder(nn.Module):
    """
    RNN (LSTM) autoencoder для входа (B, C, T).
    - Encoder: LSTM -> латент z (h_T)
    - Decoder: LSTM с пошаговой генерацией длиной T
    """
    def __init__(self, in_ch: int, hidden: int = 128, z_dim: int = 128, num_layers: int = 1, use_gru: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.hidden = hidden
        self.z_dim = z_dim
        self.num_layers = num_layers

        RNN = nn.GRU if use_gru else nn.LSTM

        # Encoder: (B, T, C) -> h_T (num_layers, B, hidden)
        self.encoder = RNN(input_size=in_ch, hidden_size=hidden, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.enc_to_z = nn.Linear(hidden, z_dim)

        # Decoder init из латента
        self.z_to_h0 = nn.Linear(z_dim, hidden * num_layers)  # разворачиваем в (num_layers, B, hidden)
        self.z_to_c0 = None if use_gru else nn.Linear(z_dim, hidden * num_layers)

        # Decoder RNN
        self.decoder = RNN(input_size=in_ch, hidden_size=hidden, num_layers=num_layers, batch_first=True, bidirectional=False)

        # Проекция скрытого состояния в выходной вектор признаков (размер C)
        self.proj = nn.Linear(hidden, in_ch)

        # Специальный токен старта декодера (learnable)
        self.start_token = nn.Parameter(torch.zeros(1, 1, in_ch))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C)
        x_seq = x.transpose(1, 2)
        out, h = self.encoder(x_seq)               # h: (num_layers, B, hidden) [+ c, если LSTM]
        if isinstance(h, tuple):                   # LSTM: (h_n, c_n)
            h_n, _ = h
        else:                                      # GRU: h_n
            h_n = h
        h_last = h_n[-1]                           # (B, hidden)
        z = self.enc_to_z(h_last)                  # (B, z_dim)
        return z

    def _init_decoder_state(self, z: torch.Tensor):
        # z: (B, z_dim) -> начальные состояния decoder’а
        B = z.size(0)
        h0 = self.z_to_h0(z).view(self.num_layers, B, self.hidden)
        if self.z_to_c0 is None:
            return h0
        c0 = self.z_to_c0(z).view(self.num_layers, B, self.hidden)
        return (h0, c0)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """ Инференс без teacher forcing. """
        self.eval()
        z = self.encode(x)
        return self.decode(z, T=x.size(-1), teacher_forcing_ratio=0.0)

    def decode(self, z: torch.Tensor, T: int, teacher_forcing_ratio: float = 0.0, y_teacher: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Декодирование длиной T.
        - teacher_forcing_ratio \in [0,1]
        - y_teacher: (B, C, T) — «правильные» сдвинутые входы для teacher forcing
        Возвращает: (B, C, T)
        """
        B = z.size(0)
        state = self._init_decoder_state(z)  # (num_layers,B,H) или tuple для LSTM

        # Стартовый вход: learnable start token, размноженный по батчу
        inp = self.start_token.expand(B, 1, self.in_ch)  # (B,1,C)

        outputs = []
        for t in range(T):
            out_t, state = self.decoder(inp, state)      # out_t: (B,1,H)
            y_t = self.proj(out_t)                       # (B,1,C)
            outputs.append(y_t)

            # решаем, чем кормить следующий шаг
            if self.training and y_teacher is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_inp = y_teacher.transpose(1, 2)[:, t:t+1, :]  # (B,1,C)
            else:
                next_inp = y_t.detach()  # собственный предыдущий прогноз
            inp = next_inp

        y_hat = torch.cat(outputs, dim=1)     # (B,T,C)
        return y_hat.transpose(1, 2).contiguous()  # -> (B,C,T)

    def forward(self, x: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Тренировочный проход с teacher forcing (по умолчанию 0.5).
        Возврат: (x_hat, z)
        """
        z = self.encode(x)
        x_hat = self.decode(z, T=x.size(-1), teacher_forcing_ratio=teacher_forcing_ratio, y_teacher=x)
        return x_hat, z

def serlu(x: torch.Tensor, alpha: float = 2.90427, lambd: float = 1.07862) -> torch.Tensor:
    """
    SERLU(x) = lambd * x                    , x >= 0
             = lambd * alpha * x * exp(x)   , x < 0
    """
    pos = torch.relu(x)  # = max(x, 0)
    neg = x - pos        # = min(x, 0)
    return lambd * (pos + alpha * neg * torch.exp(neg))

class SERLU(nn.Module):
    def __init__(self, alpha: float = 2.90427, lambd: float = 1.07862, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.lambd = lambd
        self.inplace = inplace  # флаг на будущее; тут вычисления без in-place

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return serlu(x, self.alpha, self.lambd)

class LearnableSERLU(nn.Module):
    """
    Обучаемая версия SERLU:
    y = lambda * x,  если x >= 0
    y = lambda * alpha * (exp(x) - 1), если x < 0
    где alpha и lambda -- nn.Parameter.
    """
    def __init__(self, init_alpha=1.0, init_lambda=1.0):
        super().__init__()
        # делаем их параметрами модели
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.lambd = nn.Parameter(torch.tensor(float(init_lambda)))

    def forward(self, x):
        pos = F.relu(x) * self.lambd
        neg = (torch.exp(x) - 1) * self.alpha * self.lambd
        return torch.where(x >= 0, pos, neg)

# -------- helpers --------
def reflect_pad1d_det_lr(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    if left == 0 and right == 0:
        return x
    B, C, L = x.shape
    if max(left, right) >= L:
        raise ValueError(f"pad ({left},{right}) must be < input length ({L}).")
    left_slice  = x[..., 1:left+1].flip(-1) if left  > 0 else x[..., :0]
    right_slice = x[..., L-right-1:L-1].flip(-1) if right > 0 else x[..., :0]
    return torch.cat([left_slice, x, right_slice], dim=-1)

def samepad_1d(kernel_size: int, dilation: int = 1, symmetric_if_possible: bool = True):
    span = dilation * (kernel_size - 1)
    if symmetric_if_possible and span % 2 == 0:
        p = span // 2
        return (p, p)
    left = span // 2
    right = span - left
    return (left, right)

class DetReflectConv1d(nn.Module):
    """Conv1d с детерминированным reflect-паддингом. padding: int или (left,right)."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        if isinstance(padding, tuple):
            self.pad_left, self.pad_right = int(padding[0]), int(padding[1])
        else:
            p = int(padding)
            self.pad_left, self.pad_right = p, p
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=0, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        if self.pad_left or self.pad_right:
            x = reflect_pad1d_det_lr(x, self.pad_left, self.pad_right)
        return self.conv(x)

class UpShuffle2x(nn.Module):
    """Детерминированный апсемплинг ×2 через pixel-shuffle + рефайн свёрткой."""
    def __init__(self, in_ch, out_ch, k=3, drop_last: bool = False):
        super().__init__()
        self.out_ch = out_ch
        self.drop_last = drop_last
        self.expand = nn.Conv1d(in_ch, out_ch * 2, kernel_size=1, bias=True)
        pad = samepad_1d(k, dilation=1)
        self.refine = DetReflectConv1d(out_ch, out_ch, kernel_size=k, stride=1, padding=pad)
        self.act    = nn.GELU()

    def forward(self, x):
        B, _, L = x.shape
        h = self.expand(x)                                    # (B, C*2, L)
        h = h.view(B, self.out_ch, 2, L).permute(0,1,3,2).contiguous().view(B, self.out_ch, 2*L)
        if self.drop_last:                                    # делаем 2L-1 при необходимости
            h = h[..., :-1]
        h = self.refine(h)
        return self.act(h)


class ConvAE_ShuffleDet(nn.Module):
    def __init__(self, in_ch=8, z_dim=32):
        super().__init__()
        # --- Encoder (как у тебя, детермин. reflect) ---
        self.enc = nn.Sequential(
            DetReflectConv1d(in_ch, 32,  kernel_size=5, stride=2, padding=2),  # 301 -> 151
            nn.GELU(),
            DetReflectConv1d(32,  64,   kernel_size=5, stride=2, padding=2),  # 151 -> 76
            nn.GELU(),
            DetReflectConv1d(64,  128,  kernel_size=3, stride=1, padding=1),  # 76 -> 76
            nn.GELU(),
            DetReflectConv1d(128, 256,  kernel_size=3, stride=2, padding=1),  # 76 -> 38
            nn.GELU(),
            DetReflectConv1d(256, 256,  kernel_size=3, stride=1, padding=1),  # 38 -> 38
            nn.GELU(),
            DetReflectConv1d(256, 512,  kernel_size=3, stride=2, padding=1),  # 38 -> 19
            nn.GELU(),
            DetReflectConv1d(512, 512,  kernel_size=3, stride=1, padding=1),  # 19 -> 19
            nn.GELU(),
        )

        # --- Latent ---
        self.to_z   = nn.Sequential(
            nn.LayerNorm(512 * 19),
            nn.Linear(512 * 19, z_dim)
        )
        self.from_z = nn.Sequential(
            nn.Linear(z_dim, 512 * 19),
            nn.LayerNorm(512 * 19)
        )

        # --- Decoder as Sequential (19→38→76→151→301) ---
        self.dec = nn.Sequential(
            DetReflectConv1d(512, 512, kernel_size=3, stride=1, padding=samepad_1d(3)),
            nn.GELU(),                                 # 19 -> 19

            UpShuffle2x(512, 256, k=3, drop_last=False),  # 19 -> 38
            DetReflectConv1d(256, 256, kernel_size=3, stride=1, padding=samepad_1d(3)),
            nn.GELU(),                                 # 38 -> 38

            UpShuffle2x(256, 128, k=3, drop_last=False),  # 38 -> 76
            DetReflectConv1d(128, 64, kernel_size=3, stride=1, padding=samepad_1d(3)),
            nn.GELU(),                                 # 76 -> 76 (каналы 64)

            UpShuffle2x(64,  32, k=5, drop_last=True),    # 76 -> 151 (= 2*76 - 1)
            UpShuffle2x(32,  32, k=5, drop_last=True),    # 151 -> 301 (= 2*151 - 1)

            DetReflectConv1d(32, in_ch, kernel_size=5, stride=1, padding=samepad_1d(5)),
            nn.Tanh(),                                  # выход в [-1,1] или [-0.5,0.5] с масштабом
        )

    # --- API ---
    def encode(self, x):  # x: (B, 8, 301)
        h = self.enc(x)                       # (B, 512, 19)
        z = self.to_z(h.view(x.size(0), -1))  # (B, z_dim)
        return z

    def decode(self, z):  # z: (B, z_dim)
        B = z.size(0)
        h = self.from_z(z).view(B, 512, 19)   # (B, 512, 19)
        return self.dec(h)                    # (B, in_ch, 301)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ---- новые блоки: ResBlock1D и SE1D ----

class GNActConv(nn.Module):
    """GroupNorm(1) + GELU + DetReflectConv1d (stride=1)"""
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        p = (k - 1) // 2
        # self.gn  = nn.GroupNorm(1, c_in)
        self.gn  = nn.BatchNorm1d(c_in)
        self.act = nn.GELU()
        self.cv  = DetReflectConv1d(c_in, c_out, kernel_size=k, stride=1, padding=p)
    def forward(self, x):
        return self.cv(self.act(self.gn(x)))

class ResBlock1D(nn.Module):
    """Двухслойный residual-блок поверх каналов c (stride=1)."""
    def __init__(self, c, k=3):
        super().__init__()
        self.f1 = GNActConv(c, c, k)
        self.f2 = GNActConv(c, c, k)
    def forward(self, x):
        return x + self.f2(self.f1(x))

class SE1D(nn.Module):
    """Squeeze-and-Excitation по каналам (1D)."""
    def __init__(self, c, r=8):
        super().__init__()
        c_mid = max(1, c // r)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1  = nn.Conv1d(c, c_mid, 1)
        self.act  = nn.GELU()
        self.fc2  = nn.Conv1d(c_mid, c, 1)
        self.sig  = nn.Tanh()
    def forward(self, x):
        w = self.pool(x)
        w = self.fc2(self.act(self.fc1(w)))
        w = self.sig(w)
        return x * w


class ConvAE_ResSE(nn.Module):
    """
    Улучшенный AE: детермин. reflect-свёртки, UpShuffle×2, ResBlock1D на уровнях со stride=1,
    SE1D после каждого апсемпла. Геометрия: 301 -> ... -> (512,19) -> ... -> 301.
    """
    def __init__(self, in_ch: int = 8, z_dim: int = 32):
        super().__init__()
        # ---------- Encoder ----------
        self.enc = nn.Sequential(
            DetReflectConv1d(in_ch, 32,  kernel_size=5, stride=2, padding=2),  # 301 -> 151
            nn.GELU(),

            DetReflectConv1d(32,  64,   kernel_size=5, stride=2, padding=2),  # 151 -> 76
            nn.GELU(),

            DetReflectConv1d(64,  128,  kernel_size=3, stride=1, padding=1),  # 76 -> 76
            nn.GELU(),
            ResBlock1D(128, k=3),

            DetReflectConv1d(128, 256,  kernel_size=3, stride=2, padding=1),  # 76 -> 38
            nn.GELU(),

            DetReflectConv1d(256, 256,  kernel_size=3, stride=1, padding=1),  # 38 -> 38
            nn.GELU(),
            ResBlock1D(256, k=3),

            DetReflectConv1d(256, 512,  kernel_size=3, stride=2, padding=1),  # 38 -> 19
            nn.GELU(),

            DetReflectConv1d(512, 512,  kernel_size=3, stride=1, padding=1),  # 19 -> 19
            nn.GELU(),
            ResBlock1D(512, k=3),
        )

        # ---------- Latent ----------
        self.to_z   = nn.Linear(512 * 19, z_dim)
        self.from_z = nn.Linear(z_dim, 512 * 19)

        # ---------- Decoder (19→38→76→151→301) ----------
        self.dec = nn.Sequential(
            DetReflectConv1d(512, 512, kernel_size=3, stride=1, padding=samepad_1d(3)),
            nn.GELU(),
            ResBlock1D(512, k=3),                       # уровень 19

            UpShuffle2x(512, 256, k=3, drop_last=False),# 19 -> 38
            SE1D(256),
            DetReflectConv1d(256, 256, kernel_size=3, stride=1, padding=samepad_1d(3)),
            nn.GELU(),
            ResBlock1D(256, k=3),                       # уровень 38

            UpShuffle2x(256, 128, k=3, drop_last=False),# 38 -> 76
            SE1D(128),
            DetReflectConv1d(128, 64, kernel_size=3, stride=1, padding=samepad_1d(3)),
            nn.GELU(),
            ResBlock1D(64, k=3),                        # уровень 76

            UpShuffle2x(64,  32, k=5, drop_last=True),  # 76 -> 151
            SE1D(32),

            UpShuffle2x(32,  32, k=5, drop_last=True),  # 151 -> 301
            SE1D(32),

            DetReflectConv1d(32, in_ch, kernel_size=5, stride=1, padding=samepad_1d(5)),
            nn.Tanh(),  # при нормировке к [-0.5,0.5] можно заменить на ScaledTanh(scale=0.5)
        )

    # ---------- API ----------
    def encode(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, in_ch, 301)
        h = self.enc(x)                      # (B, 512, 19)
        z = self.to_z(h.flatten(1))         # (B, z_dim)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # z: (B, z_dim)
        B = z.size(0)
        h = self.from_z(z).view(B, 512, 19) # (B, 512, 19)
        return self.dec(h)                  # (B, in_ch, 301)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        hidden_dims = [32, 32, 64, 128, 256]  # num of filters in layers
        # снижаем размерность второго пространства и повышаем размерность первого
        self.shortcut1d = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=5, stride=1, padding=2, bias=False),  # меняем только каналы
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(76)  # гарантируем совпадение длины с выходом conv1d
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        conv2d = []
        in_channels = 1  # initial value of channels
        self.shortcut2d = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 3)),  # (B,1,64,76) -> (B,1,2,3)
            nn.Conv2d(1, hidden_dims[-1], kernel_size=1, bias=False),  # -> (B,256,2,3)
            nn.BatchNorm2d(hidden_dims[-1])             # по желанию
        )
        for h_dim in hidden_dims:  # conv layers
            conv2d.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,  # num of input channels
                        out_channels=h_dim,  # num of output channels
                        kernel_size=3,
                        stride=2,  # convolution kernel step
                        padding=1,  # save shape
                    ),
                    # nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                )
            )
            in_channels = h_dim  # changing number of input channels for next iteration

        self.to_z = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 3, latent_dim)
        )

        self.conv2d = nn.Sequential(*conv2d)

    def forward(self, x):
        # identity1d = self.shortcut1d(x)
        x = self.conv1d(x)
        # x += identity1d
        x = x.unsqueeze(1)
        # identity2d = self.shortcut2d(x)
        x = self.conv2d(x)
        # x += identity2d
        x = self.to_z(x)
        return x
# class Encoder(nn.Module):
#     def __init__(self, latent_dim: int, use_pretrained: bool = False):
#         super().__init__()
#
#         # -------- 1D-часть остаётся как у тебя --------
#         self.conv1d = nn.Sequential(
#             nn.Conv1d(4, 32, kernel_size=3, stride=2, padding=1),  # 301 -> 151
#             nn.SiLU(),
#             nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1), # 151 -> 76
#             nn.SiLU(),
#             nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # 76  -> 38
#             # без активации — дальше ResNet
#         )
#
#         # -------- ResNet18 как 2D-энкодер --------
#         # Вход сюда будет (B,1,64,76) → ResNet сам сожмёт до ~ (B,512,2,3)
#         weights = torchvision.models.ResNet34_Weights.DEFAULT if use_pretrained else None
#         backbone = torchvision.models.resnet34(weights=weights)
#
#         # Первый слой под 1 канал (а не 3). Страйды/ядра стандартные.
#         backbone.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=64,
#             kernel_size=7, stride=2, padding=3, bias=False
#         )
#         # Если взяли pretrained RGB — корректно инициализируем conv1 из Среднего по каналам:
#         # with torch.no_grad():
#         #     # старые веса были на 3 канала → усредняем их
#         #     old = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).conv1.weight  # (64,3,7,7)
#         #     backbone.conv1.weight.copy_(old.mean(dim=1, keepdim=True))
#
#         # Забираем «тело» без классификатора
#         self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
#         self.layer1 = backbone.layer1
#         self.layer2 = backbone.layer2
#         self.layer3 = backbone.layer3
#         self.layer4 = backbone.layer4
#
#         # Приводим каналы 512 → 256, и фиксируем пространственный размер к (2,3),
#         # чтобы to_z совпал с твоим (256 * 2 * 3).
#         self.proj_512_to_256 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
#         self.pool_2x3 = nn.AdaptiveAvgPool2d((2, 3))
#
#         # -------- Латент --------
#         self.to_z = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 2 * 3, latent_dim)
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, 4, 301)
#         conv1d → (B, 64, 38)
#         unsqueeze → (B, 1, 64, 38)  <-- ВАЖНО: у тебя было (1,64,76) в старой версии; теперь высота=64, ширина=38.
#         Но ResNet ожидает «картинку», поэтому подаём (B,1,H,W) = (B,1,64,38).
#         """
#         # 1D часть
#         x = self.conv1d(x)          # (B, 64, 38)
#         x = x.unsqueeze(1)          # (B, 1, 64, 38)
#
#         # 2D ResNet
#         x = self.stem(x)            # (B, 64, 32, 19)
#         x = self.layer1(x)          # (B, 64, 32, 19)
#         x = self.layer2(x)          # (B, 128, 16, 10)
#         x = self.layer3(x)          # (B, 256, 8, 5)
#         x = self.layer4(x)          # (B, 512, 4, 3)  ~зависит от входной ширины, дальше усредним
#         x = self.proj_512_to_256(x) # (B, 256, H, W)
#         x = self.pool_2x3(x)        # (B, 256, 2, 3)
#
#         # в латент
#         z = self.to_z(x)            # (B, latent_dim)
#         return z

class Decoder(nn.Module):
    """
    Симметричен твоему Encoder:
      z (latent_dim)
        -> Linear -> (256, 2, 3)
        -> ConvT2d ×5: (256,2,3) -> (1,64,76)
        -> reshape to (64,76)
        -> ConvT1d ×2: (64,76) -> (4,301)
    """
    def __init__(self, latent_dim):
        super().__init__()
        # Разворачиваем латент обратно в "бутылочное горлышко" энкодера
        self.from_z = nn.Linear(latent_dim, 256 * 2 * 3)

        # ----- 2D "декодер" (инверсия conv2d-части) -----
        # Напоминание о размерах по пути энкодера:
        # (1,64,76) -> (32,32,38) -> (32,16,19) -> (64,8,10) -> (128,4,5) -> (256,2,3)
        # Обратный путь (ConvTranspose2d, stride=2, k=3, p=1) c подобранным output_padding(H,W):
        deconv2d = []
        # (256,2,3) -> (128,4,5): H: op=1, W: op=0
        deconv2d += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.SiLU(),
            # nn.BatchNorm2d(128),
        ]
        # (128,4,5) -> (64,8,10): H: op=1, W: op=1
        deconv2d += [
            nn.ConvTranspose2d(128, 64,  kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.SiLU(),
            # nn.BatchNorm2d(64),
        ]
        # (64,8,10) -> (32,16,19): H: op=1, W: op=0
        deconv2d += [
            nn.ConvTranspose2d(64,  32,  kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.SiLU(),
            # nn.BatchNorm2d(32),
        ]
        # (32,16,19) -> (32,32,38): H: op=1, W: op=1
        deconv2d += [
            nn.ConvTranspose2d(32,  32,  kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.SiLU(),
            # nn.BatchNorm2d(32),
        ]
        # (32,32,38) -> (1,64,76): H: op=1, W: op=1
        deconv2d += [
            nn.ConvTranspose2d(32,  1,   kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.SiLU(),
            # nn.BatchNorm2d(1),
            # тут BN не обязателен; оставим без активации — пойдём дальше в 1D часть
        ]
        self.deconv2d = nn.Sequential(*deconv2d)

        # ----- 1D "декодер" (инверсия conv1d-части) -----
        # После 2D части имеем (B,1,64,76) -> squeezе -> (B,64,76)
        # Дальше надо (64,76) -> (32,151) -> (4,301)
        # Для ConvTranspose1d с (k=3, s=2, p=1, output_padding=0) получаем L_out = 2*L - 1.
        self.shortcut1d = nn.Sequential(
            nn.Conv1d(64, 4, kernel_size=5, stride=1, padding=2, bias=False),  # 64->4, длина 76 сохраняется
            nn.BatchNorm1d(4),
            nn.AdaptiveAvgPool1d(301),                                         # длина -> 301
        )
        self.deconv1d = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # 76 -> 151
            nn.SiLU(inplace=True),
            nn.ConvTranspose1d(32,  12, kernel_size=3, stride=2, padding=1, output_padding=0),  # 151 -> 301
        )

        # Финальная активация по желанию:
        self.out_act = nn.Tanh()  # или nn.Tanh() / ScaledTanh(...)

    def forward(self, z):
        B = z.size(0)
        # z -> (B,256,2,3)
        h = self.from_z(z).view(B, 256, 2, 3)

        # 2D разворот до (B,1,64,76)
        h = self.deconv2d(h)

        # -> (B,64,76)
        h = h.squeeze(1)

        # 1D разворот до (B,4,301)
        # identity = self.shortcut1d(h)
        x_hat = self.deconv1d(h)
        # x_hat += identity
        x_hat = self.out_act(x_hat)
        return x_hat


class LitAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        # here is the logic how data is moved through AE
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent