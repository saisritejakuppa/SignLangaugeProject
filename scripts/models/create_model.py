from .models import UNet, Discriminator

def GetModels(opt):
    Generator = UNet(opt.input_nc, opt.output_nc)
    discriminator = Discriminator(opt.input_nc +  opt.output_nc)
    return Generator, discriminator