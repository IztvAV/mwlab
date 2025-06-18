import copy

import torch
from numpy.ma.core import identity
from torch import nn
import torch.nn.functional as F

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
        'soft_plus': F.softplus
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
        final = x * y.expand_as(x)
        return final


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='relu', use_se=False, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode="zeros")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="zeros")
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


class ResNet1DFlexible(nn.Module):
    def __init__(self, in_channels=8, out_channels=10,
                 first_conv_channels=64, first_conv_kernel=7,
                 first_maxpool_kernel=3,
                 layer_channels=[64, 128, 256, 512],
                 num_blocks=[1, 2, 3, 1],
                 activation_in='relu', activation_block='relu',
                 use_se=True, se_reduction=16):
        super().__init__()

        self.activation_name = activation_in
        self.activation = get_activation(activation_in)

        dilation = 1
        self.conv1 = nn.Conv1d(in_channels, first_conv_channels,
                               kernel_size=first_conv_kernel,
                               stride=2,
                               padding=dilation*(first_conv_kernel // 2),
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(first_conv_channels)
        # self.bn1 = nn.GroupNorm(num_groups=8, num_channels=first_conv_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=first_maxpool_kernel, stride=2, padding=1)

        self.layers = nn.ModuleList()
        in_ch = first_conv_channels
        for out_ch, n_blocks in zip(layer_channels, num_blocks):
            self.layers.append(
                self.make_layer(in_ch, out_ch, n_blocks,
                                stride=2 if in_ch != out_ch else 1,
                                activation=activation_block,
                                use_se=use_se,
                                se_reduction=se_reduction)
            )
            in_ch = out_ch

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            # nn.LayerNorm(in_ch),
            nn.Linear(in_ch, out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool1d(1)
        )

    @staticmethod
    def make_layer(in_channels, out_channels, num_blocks, stride, activation, use_se, se_reduction):
        layers = [BasicBlock1D(in_channels, out_channels, stride, activation,
                               use_se=use_se, se_reduction=se_reduction)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1,
                                       activation=activation, use_se=use_se, se_reduction=se_reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        identity = self.shortcut(x)
        identity = identity.view(identity.size(0), -1)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        if type(self.fc) != nn.Identity:
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
        # x += identity
        # x = self.activation_fun(x)
        return x


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



class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)

        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)

        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.dropout = nn.Dropout1d(0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return self.dropout(out)


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

        # --------------------------  1 conv-слой ---------------------------
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding='same')
        # --------------------------  2 conv-слой ---------------------------
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
        # --------------------------  3 conv-слой ---------------------------
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
        # --------------------------  4 conv-слой ---------------------------
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
        self.seq_conv = nn.Sequential(
            # --------------------------  1 conv-слой ---------------------------
            self.conv1,
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3, padding=1),
            # --------------------------  2 conv-слой ---------------------------
            self.conv2,
            nn.ReLU(),
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
            nn.Linear(64 * 26, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels),  # N - количество ненулевых элементов матрицы связи
            nn.Tanh()
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


class MLPMixerBlock1D(nn.Module):
    def __init__(self, num_tokens, num_channels, token_dim, channel_dim):
        super().__init__()
        # Token-mixing MLP
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(num_tokens),
            nn.Linear(num_tokens, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_tokens),
        )
        # Channel-mixing MLP
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, num_channels),
        )

    def forward(self, x):
        # x: (B, C, T)
        y = x.transpose(1, 2)  # (B, T, C)
        y = y + self.token_mixing(y)
        y = y.transpose(1, 2)  # (B, C, T)
        y = y + self.channel_mixing(y)
        return y

class MLPMixer1D(nn.Module):
    def __init__(self, input_channels, input_length, num_blocks=4,
                 token_dim=64, channel_dim=128, hidden_dim=256, output_dim=1):
        super().__init__()
        # "Patch embedding": в 1D это conv1d
        self.patch_embed = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        # Mixer blocks
        self.mixer_blocks = nn.Sequential(
            *[MLPMixerBlock1D(input_length, hidden_dim, token_dim, channel_dim)
              for _ in range(num_blocks)]
        )
        # Final head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, C, T)
        x = self.patch_embed(x)  # (B, hidden_dim, T)
        x = self.mixer_blocks(x)  # (B, hidden_dim, T)
        x = self.pool(x).squeeze(-1)  # (B, hidden_dim)
        x = self.fc(x)  # (B, output_dim)
        return x


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

        # Get final predictions
        output = self.fc_final(rnn_output.squeeze(1))
        return output

import math

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
            nn.Conv1d(channels[-2], channels[-1], 1, bias=False),
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
