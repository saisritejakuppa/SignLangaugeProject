from .models import Generator, Discriminator

def GetModels(opt):
    Generator = Generator(opt.input_nc, opt.output_nc)
    discriminator = Discriminator(opt.input_nc +  opt.output_nc)
    return Generator, discriminator