import torch
from torch.nn import functional as F


class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        # 224x224x1 => 224x224x8
        self.conv_1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=1
        )  # (1(28-1) - 28 + 3) / 2 = 1
        # 224x224x8 => 112x112x8
        self.pool_1 = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=0
        )  # (2(14-1) - 28 + 2) = 0
        # 112x112x8 => 112x112x16
        self.conv_2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1
        )  # (1(14-1) - 14 + 3) / 2 = 1
        # 112x112x16 => 56x56x16
        self.pool_2 = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=0
        )  # (2(7-1) - 14 + 2) = 0

        # 56x56x16 => 56x56x32
        self.conv_3 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
        )  # (1(14-1) - 14 + 3) / 2 = 1
        # 56x56x32 => 28x28x32
        self.pool_3 = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=0
        )  # (2(7-1) - 14 + 2) = 0

        self.linear_1 = torch.nn.Linear(28 * 28 * 32, num_classes)
        self.drop_out = torch.nn.Dropout2d(0.2)

    def forward(self, x):
        # x = x.type(torch.float)

        out = self.conv_1(x)
        out = F.relu(out)
        out = self.pool_1(out)
        out = self.drop_out(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.pool_2(out)
        out = self.drop_out(out)

        out = self.conv_3(out)
        out = F.relu(out)
        out = self.pool_3(out)
        out = self.drop_out(out)

        logits = self.linear_1(out.view(-1, 28 * 28 * 32))
        probas = F.sigmoid(logits)
        return probas


# class Resnet(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(Resnet, self).__init__()
#         self.num_classes = num_classes

#         self.model = models.resnet18(pretrained=False)
#         self.model.conv1 = torch.nn.Conv2d(
#             1, 64, kernel_size=7, stride=2, padding=3, bias=False
#         )
#         fc = torch.nn.Sequential(
#             torch.nn.Linear()
#         )
