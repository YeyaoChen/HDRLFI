import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        vgg_pretrained_features = models.vgg19(pretrained=True).features
        # print(vgg_pretrained_features)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 32):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False
            for param in self.slice2.parameters():
                param.requires_grad = False
            for param in self.slice3.parameters():
                param.requires_grad = False
            for param in self.slice4.parameters():
                param.requires_grad = False
            for param in self.slice5.parameters():
                param.requires_grad = False

    def forward(self, in_X):
        # input [b,c,h,w]
        in_X = self.slice1(in_X)
        h_relu1_2 = in_X
        in_X = self.slice2(in_X)
        h_relu2_2 = in_X
        in_X = self.slice3(in_X)
        h_relu3_2 = in_X
        in_X = self.slice4(in_X)
        h_relu4_2 = in_X
        in_X = self.slice5(in_X)
        h_relu5_2 = in_X
        h_relu = [h_relu1_2, h_relu2_2, h_relu3_2, h_relu4_2, h_relu5_2]
        return h_relu


if __name__ == '__main__':
    vgg19 = Vgg19(requires_grad=False)




