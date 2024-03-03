import torch
import torch.nn as nn
import torch.nn.functional as F

# Assume the input time-series data has a shape of [batch_size, channels, length]
# For example: [batch_size, 1, sequence_length]
# You need to adjust 'sequence_length' to fit your actual data
sequence_length = 17  # Example length, you need to set this value based on your data
calculated_length = (sequence_length - 3*4)  # This calculation is based on the conv layers with kernel size 10 and stride 1, without padding.


num_features = 3 # DO GO PEAK
kernel_size = 3
sequence_length = 17

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Define the 1D convolutional layers
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1)

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=128 * calculated_length, out_features=384)
        self.fc2 = nn.Linear(in_features=384, out_features=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Define the fully connected layer
        self.fc = nn.Linear(in_features=5, out_features=384)

        # Define the 1D deconvolutional layers
        self.deconv1 = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=1)
        self.deconv3 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=kernel_size, stride=1)
        self.deconv4 = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        # Reshape the output to fit the deconvolutional layers
        x = x.view(x.size(0), 128, 3)  # Adjust the reshaping depending on the actual output size
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        return x

class DCAE(nn.Module):
    def __init__(self):
        super(DCAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Note: 'calculated_length' needs to be calculated based on the length of the time-series after the convolutions
# You will need to adjust this value based on your specific input data size and the expected output size

# Instantiate the model
dcae = DCAE()


# Example dummy input
dummy_input = torch.randn(1000, 3, sequence_length)
output = dcae(dummy_input)

print(output.shape)
