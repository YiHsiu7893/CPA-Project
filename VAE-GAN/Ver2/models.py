import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder,self).__init__()
    self.conv1=nn.Conv2d(1, 8, kernel_size=1, padding=1)
    self.conv2=nn.Conv2d(8, 16, kernel_size=3, padding=1)
    self.conv3=nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.conv4=nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv5=nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.conv6=nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.conv7=nn.Conv2d(256, 512, kernel_size=3, padding=1)
    
    self.relu=nn.LeakyReLU(inplace=True)
    self.downsample=nn.MaxPool2d(kernel_size=2)
    
    self.fc_mean=nn.Linear(512,128)
    self.fc_logvar=nn.Linear(512,128)   #latent dim=128
  
  def forward(self,x,epoch):
    batch_size=x.size()[0]
    
    if epoch>=11:
        x=self.relu(self.conv2(self.relu(self.conv1(x))))
        x=self.downsample(x)
    
    if epoch>=9:
        x=self.downsample(self.relu(self.conv3(x)))
    
    if epoch>=7:
        x=self.downsample(self.relu(self.conv4(x)))
    
    if epoch>=5:
        x=self.downsample(self.relu(self.conv5(x)))
    
    if epoch>=3:
        x=self.downsample(self.relu(self.conv6(x)))
    
    if epoch>=1:
        x=self.downsample(self.relu(self.conv7(x)))  
    
    mean=self.fc_mean(x)
    logvar=self.fc_logvar(x)
    
    return mean,logvar
    
    
class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    self.linear=nn.Linear(128,4*4*512)
    
    self.upsample1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
    self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
    self.upsample2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
    self.upsample3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
    self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    self.upsample4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
    self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
    self.upsample5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
    self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
    self.upsample6 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
    self.conv6 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
    self.conv7 = nn.Conv2d(8, 1, kernel_size=1, padding=1)
    
    self.relu=nn.LeakyReLU(inplace=True)
    self.tanh=nn.Tanh()

  def forward(self,x,epoch):
    batch_size=x.size()[0]
    
    x=self.relu(self.linear(x))
    
    if epoch>=1:
        x=self.relu(self.conv1(self.upsample1(x)))
        
        if epoch>=3:
            x=self.relu(self.conv2(self.upsample2(x)))
            
            if epoch>=5:
                x=self.relu(self.conv3(self.upsample3(x)))
                
                if epoch>=7:
                    x=self.relu(self.conv4(self.upsample4(x)))
                    
                    if epoch>=9:
                        x=self.relu(self.conv5(self.upsample5(x)))
                        
                        if epoch>=11:
                            x=self.relu(self.conv6(self.upsample6(x)))
                            x=self.tanh(self.conv7(x))
    
    return x
    
    
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()  
    self.conv1=nn.Conv2d(1, 8, kernel_size=1, padding=1)
    self.conv2=nn.Conv2d(8, 16, kernel_size=3, padding=1)
    self.conv3=nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.conv4=nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv5=nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.conv6=nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.conv7=nn.Conv2d(256, 512, kernel_size=3, padding=1)
    
    self.relu=nn.LeakyReLU(inplace=True)
    self.downsample=nn.MaxPool2d(kernel_size=2)
    self.fc=nn.Linear(512,1)
    self.sigmoid=nn.Sigmoid()

  def forward(self,x,epoch):
    batch_size=x.size()[0]
    
    if epoch>=11:
        x=self.relu(self.conv2(self.relu(self.conv1(x))))
        x=self.downsample(x)
    
    if epoch>=9:
        x=self.downsample(self.relu(self.conv3(x)))
    
    if epoch>=7:
        x=self.downsample(self.relu(self.conv4(x)))
    
    if epoch>=5:
        x=self.downsample(self.relu(self.conv5(x)))
    
    if epoch>=3:
        x=self.downsample(self.relu(self.conv6(x)))
    
    if epoch>=1:
        x=self.downsample(self.relu(self.conv7(x)))
    
    x1=x
    x=self.sigmoid(self.fc(x))

    return x,x1
    
    
class VAE_GAN(nn.Module):
  def __init__(self):
    super(VAE_GAN,self).__init__()
    self.encoder=Encoder()
    self.generator=Generator()
    self.discriminator=Discriminator()
    
    self.encoder.apply(weights_init)
    self.generator.apply(weights_init)
    self.discriminator.apply(weights_init)


  def forward(self,x,epoch):
    bs=x.size()[0]
    z_mean,z_logvar=self.encoder(x,epoch)
    std = z_logvar.mul(0.5).exp_()
        
    #sampling epsilon from normal distribution
    epsilon = torch.randn(bs, 128, device=device)
    z=z_mean+std*epsilon
    x_tilda=self.generator(z,epoch)
      
    return z_mean,z_logvar,x_tilda
    
    
