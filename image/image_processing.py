import torch
import torch.nn as nn

class ImageGradientModule(nn.Module):
    def __init__(self, channels, device, dtype, reflect_padding=True):
        super(ImageGradientModule, self).__init__()

        # Scharr kernel
        kernel_x = (1.0/32.0) * torch.tensor( [ [ -3.0, 0.0,  3.0],
                                                [-10.0, 0.0, 10.0],
                                                [ -3.0, 0.0,  3.0] ], requires_grad=False, device=device, dtype=dtype)
        kernel_x = kernel_x.view((1,1,3,3))
        self.kernel_x = kernel_x.repeat(channels,1,1,1)

        kernel_y = (1.0/32.0) * torch.tensor( [ [-3.0, -10.0, -3.0],
                                                [ 0.0,   0.0,  0.0],
                                                [ 3.0,  10.0,  3.0] ], requires_grad=False, device=device, dtype=dtype)
        kernel_y = kernel_y.view((1,1,3,3))
        self.kernel_y = kernel_y.repeat(channels,1,1,1)

        self.reflect_padding = reflect_padding

    def forward(self, x):
        gx =  nn.functional.conv2d(nn.functional.pad(x, (1,1,1,1), mode='reflect' if self.reflect_padding else 'constant'),
                                   self.kernel_x, groups=x.shape[1])

        gy =  nn.functional.conv2d(nn.functional.pad(x, (1,1,1,1), mode='reflect' if self.reflect_padding else 'constant'),
                                   self.kernel_y, groups=x.shape[1])

        return gx, gy



def get_image_grad(image):
    image = image.unsqueeze(0)
    channels = image.shape[1]
    device = image.device
    dtype = image.dtype

    grad_module = ImageGradientModule(channels, device, dtype, reflect_padding=False)

    gx, gy = grad_module(image)
    img_grads = torch.stack((gx, gy), dim=2)
    img_grads = img_grads.squeeze(0)
    return img_grads