import torch
import torch.nn as nn


# c_in is patch length (16
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=1,
                                   padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


def main():
    c_in = 16  # Number of input channels
    d_model = 32  # Number of output channels (32)
    sequence_length = 10
    batch_size = 1

    # Instantiate the model
    model = TokenEmbedding(c_in=c_in, d_model=d_model)

    # Generate a random input tensor with shape (batch_size, sequence_length, c_in)
    input_tensor = torch.randn(batch_size, sequence_length, c_in)

    # Pass the tensor through the model
    print(input_tensor.shape)
    output_tensor = model(input_tensor)
    print(output_tensor)
    print(output_tensor.shape)


if __name__ == "__main__":
    main()