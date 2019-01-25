from src.connect4.utils import Connect4Stats as info

import pickle
import torch.nn as nn

data = pickle.load(open('/home/richard/Downloads/connect4.pkl', 'rb'))

# Input with N * 3 * (6,7)
# Output with N * 16 * (6,7)
convolutional_layer = \
    nn.Sequential(nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            dilation=1,
                            groups=1,
                            bias=False),
                  nn.BatchNorm2d(16),
                  nn.ReLU())


# Input with N * 16 * (6,7)
# Output with N * 16 * (6,7)
class ResidualLayer(nn.Module):
    def __init__(self):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(16)
        self.reul = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.reul(out)

        out += residual
        out = self.reul(out)
        return out


# Input with N * 16 * (6,7)
# Output with N * 1
class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(16, 4, 1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.reul = nn.ReLU()
        self.fc1 = nn.Linear(4 * info.area, 4 * info.area)
        self.fc2 = nn.Linear(4 * info.area, 1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.batch_norm(x)
            x = self.relu(x)
            for _ in range(4):
                x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x


net = nn.Sequential(convolutional_layer,
                    ResidualLayer(),
                    ResidualLayer(),
                    ValueHead())

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
# import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs
#         inputs, labels = data

#         # RW added
#         inputs, labels = inputs.to(device), labels.to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')
