import torch
import torch.nn as nn
import torch.nn.functional as F


class DCAE(nn.Module):
    def __init__(self, sequence_length=17, num_features=3, num_kernels=(32, 64, 128, 128), kernel_size=3):
        super(DCAE, self).__init__()

        self.sequence_length = sequence_length
        self.num_features = num_features

        # Encoder - 1D Convolutional layers
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(self.num_features, num_kernels[0], kernel_size),
            nn.ReLU(),
            nn.Conv1d(num_kernels[0], num_kernels[1], kernel_size),
            nn.ReLU(),
            nn.Conv1d(num_kernels[1], num_kernels[2], kernel_size),
            nn.ReLU(),
            nn.Conv1d(num_kernels[2], num_kernels[3], kernel_size),
            nn.ReLU()
        )

        # Calculate the output size after the convolutional layers
        self.conv_output_size = self._calculate_conv_output_size()

        # Fully connected layers for the encoder
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 384),
            nn.ReLU(),
            nn.Linear(384, 5),
            nn.ReLU()
        )

        # Fully connected layers for the decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(5, 384),
            nn.ReLU(),
            nn.Linear(384, self.conv_output_size),
            nn.ReLU()
        )

        # 1D Deconvolutional layers for the decoder
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose1d(num_kernels[3], num_kernels[2], kernel_size),
            nn.ReLU(),
            nn.ConvTranspose1d(num_kernels[2], num_kernels[1], kernel_size),
            nn.ReLU(),
            nn.ConvTranspose1d(num_kernels[1], num_kernels[0], kernel_size),
            nn.ReLU(),
            nn.ConvTranspose1d(num_kernels[0], self.num_features, kernel_size),
            nn.ReLU()
        )

    def forward(self, x):
        # Encode
        x = self.encoder_conv(x)
        x = self.encoder_fc(x)

        # Decode
        x = self.decoder_fc(x)
        x = x.view(x.size(0), -1, self.sequence_length)  # Reshape to the original sequence length
        x = self.decoder_deconv(x)
        return x

    def _calculate_conv_output_size(self):
        # Dummy calculation for convolution output size. Needs to be calculated based on your model's architecture.
        size = self.sequence_length
        for _ in range(4):  # Assuming 4 convolutional layers
            size = size - (kernel_size - 1)  # Assuming stride of 1 and no padding
        return size * num_kernels[-1]  # Assuming the last conv layer's number of kernels


# Instantiate the model
model = DCAE()

# View the model architecture
print(model)
#
# # Example usage
# sequence_length = 17  # The number of time points per sequence
# num_features = 3  # The number of features per time point (DO, GO, PEAK)
# model = DCAE(sequence_length, num_features)
#
# print(model)
