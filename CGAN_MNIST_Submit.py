import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, z_size, img_size_x, img_size_y,class_num, class_emb, generator_channels):
        super().__init__()
        self.z_size = z_size
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.condition = nn.Embedding(num_embeddings=class_num, embedding_dim=class_emb)
        
        self.input_layer = nn.Sequential(
            nn.Linear(self.z_size + class_emb, generator_channels[0]*7*7),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=generator_channels[0],out_channels=generator_channels[1],kernel_size=(3,3),stride=(1,1),padding=(1,1), bias=False),
            nn.BatchNorm2d(generator_channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=generator_channels[1],out_channels=generator_channels[2],kernel_size=(3,3),stride=(2,2),padding=(1,1), bias=False),
            nn.BatchNorm2d(generator_channels[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=generator_channels[2],out_channels=generator_channels[3],kernel_size=(3,3),stride=(2,2),padding=(1,1), bias=False),
            nn.BatchNorm2d(generator_channels[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=generator_channels[3],out_channels=generator_channels[4],kernel_size=(4,4),stride=(1,1),padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, z, label):
        # Transform label into embedding 
        condn = self.condition(label)
        # Prepare input to generator
        x = torch.cat([z, condn], dim=1)
        x = self.input_layer(x)
        x = x.view(-1, generator_channels[0], 7, 7)
        x = self.deconv(x)
        # Crop the image to desired output size
        x = x[:,:,:img_size_x,:img_size_y]
        return(x)
    

class Discriminator(nn.Module):
    def __init__(self, img_size_x, img_size_y,class_num, class_emb, discriminator_channels) -> None:
        super().__init__(), 
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.condition = nn.Embedding(num_embeddings=class_num, embedding_dim=class_emb)
        self.stride = 2
        self.c_size = [int(self.img_size_x/(self.stride**2)),int(self.img_size_y/self.stride**2)] 

        self.conv = nn.Sequential(
            nn.Conv2d(1, discriminator_channels[0], kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(discriminator_channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_channels[0], discriminator_channels[1], kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(discriminator_channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
        )


        self.classifier = nn.Sequential(
            nn.Linear(discriminator_channels[1] * self.c_size[0] * self.c_size[1] + class_emb, discriminator_channels[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_channels[2], discriminator_channels[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_channels[3], discriminator_channels[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_channels[4], 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        condn = self.condition(label)
        x = self.conv(x)
        # print(x.shape)
        x = x.view(-1, discriminator_channels[1] * self.c_size[0] * self.c_size[1])
        # print(x.shape)
        x = torch.cat([x,condn], dim=1)
        x = self.classifier(x)
        return x


def train_one_step(batch_size, generator, discriminator, optimizerG, optimizerD, criterion, real_images, real_labels):
    
    loss_val = {
        "generator": 0,
        "discriminator": 0
    }

    loss = 0
    optimizerG.zero_grad()
    z = torch.randn(batch_size, z_size).to(device)
    fake_labels = torch.tensor(np.random.randint(0, class_num, batch_size)).to(device)
    fake_images = generator(z, fake_labels)
    predictions = discriminator(fake_images, fake_labels)
    targets = torch.Tensor(np.ones(batch_size)).unsqueeze(dim=1).to(device)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizerG.step()
    loss_val['generator'] = loss.item()

    loss = 0
    optimizerD.zero_grad()
    predictions = discriminator(real_images, real_labels)
    loss += criterion(predictions, targets)
    z = torch.randn(batch_size, z_size).to(device)
    fake_labels = torch.tensor(np.random.randint(0, class_num, batch_size)).to(device)
    fake_images = generator(z, fake_labels)
    predictions = discriminator(fake_images, fake_labels)
    targets = torch.Tensor(np.zeros(batch_size)).unsqueeze(dim=1).to(device)
    loss += criterion(predictions, targets)
    loss.backward()
    optimizerD.step()
    loss_val['discriminator'] = loss.item()

    return(loss_val)

if __name__ == "__main__":

    train = False    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('torch version:',torch.__version__)
    print('device:', device)

    img_size_x = 28 
    img_size_y = 28
    batch_size = 128
    class_emb = 5

    # Dimension of latent vector
    z_size = 100
    epochs = 50  # Train epochs
    lr = 1e-4

    generator_channels = [128, 64, 32, 16, 1]
    discriminator_channels = [32, 16, 512, 256, 128]

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    class_list = train_data.classes
    class_num = len(class_list)

    train_loader = DataLoader(
        dataset= train_data,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True)

    discriminator = Discriminator(img_size_x, img_size_y, class_num, class_emb, discriminator_channels).to(device)
    generator = Generator(z_size, img_size_x, img_size_y, class_num, class_emb, generator_channels).to(device)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)

    losses_plot = np.zeros([2, epochs])

    try:
        if(train):
            for epoch in range(epochs):
                for batch, (X, y) in enumerate(train_loader):
                    X = X.to(device)
                    y = y.to(device)

                    generator.train()
                    discriminator.train()
                    loss_val = train_one_step(batch_size, generator, discriminator, optimizerG, optimizerD, criterion, X, y)
                    losses_plot[1, epoch] += loss_val['discriminator']/len(train_loader)
                    losses_plot[0, epoch] += loss_val['generator']/len(train_loader)
                print(f"Epoch {epoch}, lossG={losses_plot[0, epoch]}, lossD={losses_plot[1, epoch]}")
            torch.save(generator.state_dict(), "generator_self_conv_submit.pt")
            torch.save(discriminator.state_dict(), "discriminator_self_conv_submit.pt")
        elif(not train):
            generator.load_state_dict(torch.load("generator_self_conv_submit.pt"))
            discriminator.load_state_dict(torch.load("discriminator_self_conv_submit.pt"))
    except Exception:
        print(f"Training terminated after {epoch} epochs")
    finally:
        n_examples = 5
        noise = torch.randn(n_examples*class_num, z_size).to(device)
        # labels = torch.tensor(np.arange(0, class_num, 1)).to(device)
        labels = torch.LongTensor([i for _ in range(n_examples) for i in range(class_num)]).to(device)
        generator.eval()
        img = generator(noise, labels).cpu().detach().squeeze().numpy()
        labels = np.arange(0, class_num, 1)
        fig, ax = plt.subplots(n_examples, class_num)
        for i in range(class_num):
            for j in range(n_examples):
                ax[j, i].imshow(img[j*class_num + i], cmap='gray')
                ax[j, i].set_title(str(class_list[labels[i]]))
        plt.show()
        plt.close()

        if(train):
            plt.plot(np.arange(0, epochs, 1), losses_plot[0, :])
            plt.plot(np.arange(0, epochs, 1), losses_plot[1, :])
            plt.legend(["Generator Loss", "Discriminator Loss"])
            plt.title("Training Losses")
            plt.xlabel("Epochs")
            plt.show()
        
