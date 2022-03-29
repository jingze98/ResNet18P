from InferImagenetResNet import InferImagenetResNet
import torch

visual = InferImagenetResNet(block_name="BasicBlock",
                             layers=[2, 2, 2, 2],
                             xchannels=[3, 64, 25, 64, 38, 19, 128, 128, 38, 38, 256, 256, 256, 256, 512, 512, 512, 512],
                             xblocks=[1, 1, 2, 2],
                             deep_stem=0,
                             num_classes=512,
                             zero_init_residual=1)

