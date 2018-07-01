import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim

batch = 72
epochs = 101
learning_rate = 2e-4
nz = 200
cuda = True
save_models = 1

transform = transforms.Compose(
        [transforms.ToTensor()])

imageset = torchvision.datasets.ImageFolder("special", transform=transform)

trainloader = torch.utils.data.DataLoader(imageset, batch_size = batch ,
                                          shuffle=True, num_workers=2)
device = torch.device("cuda")

import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                )
        self.last_conv = nn.Conv2d(1024, 1, 4, 1, 0, bias=False)
    def forward(self, x):
        output = self.main(x)
        llayer = F.sigmoid(self.last_conv(output)).view(-1, 1).squeeze(1)
        return output, llayer

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(200, 1024, 4, 1, 0, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                nn.Tanh()
                )
    def forward(self, x):
        output = self.main(x)
        return output

class Encoder(nn.Module):
    def  __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 512, 4, 2, 1, bias=False)
                )
        self.fc1 = nn.Linear(2*2*512, 200)
        self.fc2 = nn.Linear(2*2*512, 200)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def forward(self, x):
        y =  self.conv_layers(x).view(-1, 2*2*512)
        mu, logvar = self.fc1(y), self.fc2(y)
        z = self.reparameterize(mu, logvar).view(-1, 200)
        return z.view(x.size()[0], 200, 1, 1), mu, logvar


netD = Discriminator()
netG = Generator()
enc = Encoder()

test_img = imageset[0][0].view(1, 3, 128, 128)
test_disc = netD(test_img)
print(test_disc[0].size(), test_disc[1].size())
test_noise = torch.randn(36, 200, 1, 1)
test_pics = netG(test_noise)
print(test_pics.size())
test_encode = enc(test_img)
print(test_encode[0].size())

if cuda:
    print("loading models")
    device = torch.device("cuda")
    netD = nn.DataParallel(netD)
    netG = nn.DataParallel(netG)
    enc = nn.DataParallel(enc)
    netD.to(device)
    netG.to(device)
    enc.to(device)
    print("models loaded")

criterion = nn.BCELoss()
mse_criterion = nn.MSELoss()

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=0.05*learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerE = optim.Adam(enc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

def kll(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def mseloss(v1, v2):
    return torch.sum((v2 - v1) ** 2) / v1.data.nelement()

for epoch in range(epochs):
    for i, (data, _) in enumerate(trainloader, 0):
        netD.zero_grad()
        real_imgs = data.to(device)
        batch_size = real_imgs.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        l1, output = netD(real_imgs)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        _, output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        _, output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
#        optimizerG.step()

        enc.zero_grad()
        output, mu, logvar = enc(real_imgs)
#        print(output.size())
        recon = netG(output)
        l2, _ = netD(recon)
        kll_loss = kll(mu, logvar)
        mse_loss = mse_criterion(l2, l1.detach())
        kll_loss.backward(retain_graph=True)
        mse_loss.backward()
        optimizerG.step()
        optimizerE.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, epochs, i, len(trainloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    vutils.save_image(real_imgs,
            '%s/real_samples.png' % "results",
            normalize=True)
    fake = netG(test_noise)
    vutils.save_image(fake.detach(),
            '%s/fake_samples_epoch_%03d.png' % ("results", epoch),
            normalize=True)
    if epoch % save_models == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ("models", epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ("models", epoch))
        torch.save(enc.state_dict(), '%s/enc_epoch_%d.pth' %("models", epoch))
