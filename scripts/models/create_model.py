from .models import Generator, Discriminator

def GetModels(opt):
    gen = Generator(opt.input_nc, opt.output_nc)
    disc = Discriminator(opt.input_nc +  opt.output_nc)
    return gen, disc