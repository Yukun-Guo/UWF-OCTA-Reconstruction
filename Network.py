"""
Neural network architectures for image segmentation.

This module implements custom CNN architectures based on ResNet blocks with
U-Net-style skip connections for semantic segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from torchsummary import summary

class CNNNet(nn.Module):
    def __init__(self, decoder_channels=(256, 128, 96, 64, 48)):
        """
        Initialize the CNN network.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output classes
            out_activation (str or None): Output activation function
        """
        super().__init__()
        self.octa_encoder = get_encoder('resnet50', in_channels=1, weights=None) 
                
        self.hdocta_decoder = UnetDecoder(
            encoder_channels=self.octa_encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_norm="batchnorm",
            add_center_block=False,
            attention_type=None,
            interpolation_mode="nearest",
        )
        self.hdocta_head = nn.Conv2d(48, 1, kernel_size=3, padding=1)
        
        
        
    def forward(self, octa: torch.Tensor) -> torch.Tensor:
        # Extract features from both encoders
        octa_features = self.octa_encoder(octa)  # list of feature maps at different scales
        # Branch 2: HDOCTA decoder (from OCTA features only)
        hdocta_decoder_output = self.hdocta_decoder(octa_features)
        hdocta_output = self.hdocta_head(hdocta_decoder_output)

        return hdocta_output

    def __str__(self):
        return 'CNNNet'


if __name__ == '__main__':
    # model = CNNNet(1, 3,out_activation=None)
    oct = torch.randn(1, 1, 304, 304, requires_grad=True)
    octa = torch.randn(1, 1, 304, 304, requires_grad=True)
    
    net = CNNNet(decoder_channels=(256, 128, 96, 64, 48))
    hdocta_out = net(octa)
    print(f"hdocta_out shape: {hdocta_out.shape}")
    
    # Export the model
    net.eval()  # Set to evaluation mode before export
    torch_out = net(octa)
    torch.onnx.export(net,  # model being run
                      octa,  # model input (or a tuple for multiple inputs)
                      # where to save the model (can be a file or file-like object)
                      "CNNNet.onnx",
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,  # use a more recent opset version
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['octa'],  # the model's input names
                      output_names=['hdocta_output'],  # the model's output names
                      dynamic_axes={
                          'octa': {0: 'batch_size'},
                          'hdocta_output': {0: 'batch_size'}
                      },
                      dynamo=False)  # Use legacy TorchScript exporter

    # print model summary
    # Note: torchsummary doesn't support multiple inputs well, so we skip it
    # or use torchinfo instead which handles multiple inputs better
    print("ONNX model exported successfully to CNNNet.onnx")