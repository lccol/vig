import torch

from models.vig import ViG, PyramidViG

if __name__ == '__main__':
    images = torch.rand((4, 3, 224, 224))

    out_channels = [128, 256, 512]
    # base vig
    # model = ViG(3,
    #             out_channels,
    #             heads=2,
    #             n_classes=10,
    #             input_resolution=(224, 224),
    #             reduce_factor=4)
    # pyramid vig
    model = PyramidViG(3,
                out_channels,
                heads=2,
                n_classes=10,
                input_resolution=(224, 224),
                reduce_factor=4)
    
    res = model(images)
    print(f'output shape: {res.shape}')