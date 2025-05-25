import torch
from torch import nn
import torch.nn.functional as F

from filters import MWFilter


def _get_activation(name):
    """Возвращает функцию активации по имени"""
    activations = {
        'relu': F.relu,
        'leaky_relu': F.leaky_relu,
        'elu': F.elu,
        'selu': F.selu,
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'swish': lambda x: x * torch.sigmoid(x),
        'mish': lambda x: x * torch.tanh(F.softplus(x)),
        'gelu': F.gelu,
        'none': lambda x: x
    }
    return activations.get(name.lower(), F.relu)


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


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

        # Выбор функции активации
        self.activation = _get_activation(activation)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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
                 layer_channels=[64, 128, 256, 512],
                 num_blocks=[1, 2, 3, 1],
                 activation_in='relu', activation_block='relu'):
        super().__init__()

        # Сохраняем параметр активации
        self.activation_name = activation_in
        self.activation = _get_activation(activation_in)

        # Первый сверточный слой
        self.conv1 = nn.Conv1d(in_channels, first_conv_channels,
                               kernel_size=first_conv_kernel,
                               stride=2,
                               padding=first_conv_kernel // 2)
        self.bn1 = nn.BatchNorm1d(first_conv_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Создаем слои ResNet
        self.layers = nn.ModuleList()
        in_ch = first_conv_channels
        for out_ch, n_blocks in zip(layer_channels, num_blocks):
            self.layers.append(self.make_layer(in_ch, out_ch, n_blocks,
                                               stride=2 if in_ch != out_ch else 1,
                                               activation=activation_block))
            in_ch = out_ch

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, out_channels)

    @staticmethod
    def make_layer(in_channels, out_channels, num_blocks, stride, activation):
        layers = [BasicBlock1D(in_channels, out_channels, stride, activation)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1, activation=activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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


class DenseBlock1D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(in_channels + i * growth_rate),
                nn.ReLU(),
                nn.Conv1d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class DenseNet1D(nn.Module):
    def __init__(self, in_channels=8, growth_rate=32, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.block1 = DenseBlock1D(64, growth_rate, 6)
        self.transition = nn.Sequential(
            nn.BatchNorm1d(64 + 6 * growth_rate),
            nn.Conv1d(64 + 6 * growth_rate, 128, kernel_size=1)
        )
        self.block2 = DenseBlock1D(128, growth_rate, 12)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 + 12 * growth_rate, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.transition(x)
        x = self.block2(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))


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


class ResNetRNN1D(nn.Module):
    def __init__(self, in_channels=8, out_channels=10,
                 resnet_hidden_size=512, rnn_hidden_size=256,
                 rnn_layers=1, rnn_type='LSTM', bidirectional=False):
        super().__init__()

        # ResNet part
        self.resnet = ResNet1D(in_channels, resnet_hidden_size)

        # RNN part
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.rnn_hidden_size = rnn_hidden_size

        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=out_channels,  # Using ResNet output as RNN input
                hidden_size=rnn_hidden_size,
                num_layers=rnn_layers,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=out_channels,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_layers,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be either 'LSTM' or 'GRU'")

        # Final fully connected layer
        direction_multiplier = 2 if bidirectional else 1
        self.fc_final = nn.Linear(rnn_hidden_size * direction_multiplier, out_channels)

    def forward(self, x):
        # Get initial predictions from ResNet
        resnet_output = self.resnet(x)  # shape: (batch_size, out_channels)

        # Prepare for RNN - add sequence dimension (sequence length = 1)
        # We're treating each ResNet output as a single timestep
        rnn_input = resnet_output.unsqueeze(1)  # shape: (batch_size, 1, out_channels)

        # RNN processing
        if self.rnn_type.upper() == 'LSTM':
            h0 = torch.zeros(self.rnn_layers * (2 if self.bidirectional else 1),
                             rnn_input.size(0), self.rnn_hidden_size).to(x.device)
            c0 = torch.zeros(self.rnn_layers * (2 if self.bidirectional else 1),
                             rnn_input.size(0), self.rnn_hidden_size).to(x.device)
            rnn_output, _ = self.rnn(rnn_input, (h0, c0))
        else:
            h0 = torch.zeros(self.rnn_layers * (2 if self.bidirectional else 1),
                             rnn_input.size(0), self.rnn_hidden_size).to(x.device)
            rnn_output, _ = self.rnn(rnn_input, h0)

        # Get final predictions
        output = self.fc_final(rnn_output.squeeze(1))
        return output


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)


class SENet1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, block_config=[2, 2, 2],
                 init_channels=64, reduction=16, dropout=0.2):
        super().__init__()

        # Initial layers
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, init_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers
        self.layers = nn.ModuleList()
        channels = init_channels

        for i, num_blocks in enumerate(block_config):
            out_channels = channels * (2 if i > 0 else 1)  # Double channels after first layer
            self.layers.append(self._make_layer(channels, out_channels, num_blocks,
                                                stride=1 if i == 0 else 2,
                                                reduction=reduction))
            channels = out_channels

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, reduction):
        layers = []
        layers.append(SEBasicBlock1D(in_channels, out_channels, stride, reduction))
        for _ in range(1, blocks):
            layers.append(SEBasicBlock1D(out_channels, out_channels, 1, reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)  # [batch, init_channels, L//4]

        for layer in self.layers:
            x = layer(x)  # После каждого слоя: [batch, channels*, L*]

        x = self.avgpool(x)  # [batch, channels, 1]
        x = torch.flatten(x, 1)  # [batch, channels]
        x = self.dropout(x)
        x = self.fc(x)  # [batch, num_classes]
        return x