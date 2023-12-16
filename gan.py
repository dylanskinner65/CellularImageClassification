from torch.utils.data import DataLoader, dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import image_dataset
import torch.optim as optim
from torchvision import transforms, models
from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse

# define the generator network


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return torch.tanh(self.fc2(self.relu(self.fc1(x))))

# define the discriminator network


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


def generate_images(generator, device, images_to_generate, latent_size, image_size, path):
    with torch.no_grad():
        noise = torch.randn(images_to_generate, latent_size)
        generated_images = generator(
            noise.to(device)).view(-1, 1, image_size, image_size)
        save_image(generated_images, path)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    device = args.device
    epochs = args.epochs

    # set manual seed
    torch.manual_seed(42)

    latent_size = 100
    hidden_size = 256
    image_size = 512
    lr = .00001
    batch_size = 32
    images_to_generate = 24
    checkpoint_num = 100
    gen_weight = 0.5
    disc_weight = 1
    gen_lr = 0.00002
    disc_lr = 0.00001

    dataset = image_dataset.ImageDataset(transform=transforms.ToTensor())

    generator = Generator(latent_size, hidden_size,
                          image_size * image_size).to(device)
    discriminator = Discriminator(
        image_size * image_size, hidden_size, 1).to(device)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    criterion = nn.BCELoss()
    gen_optim = optim.Adam(generator.parameters(), gen_lr)
    disc_optim = optim.Adam(discriminator.parameters(), disc_lr)

    # save losses
    disc_losses = []
    gen_losses = []

    # load checkpoint if it exists
    try:
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        gen_optim.load_state_dict(
            checkpoint['generator_optimizer_state_dict'])
        disc_optim.load_state_dict(
            checkpoint['discriminator_optimizer_state_dict'])
        print(f'Loaded checkpoint')
    except Exception as e:
        print('No checkpoint found')
        print(e)


    loop = tqdm(total=len(dataloader), position=0, leave=False)
    for epoch in range(epochs):
        for i, (real_image, _) in enumerate(dataloader):
            batch_size = real_image.size(0)

            real_image = real_image.view(-1,
                                         image_size * image_size).to(device)

            noise = torch.randn(batch_size, latent_size).to(device)

            # train the discriminator
            disc_optim.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)

            real_outputs = discriminator(real_image)
            real_loss = criterion(real_outputs, real_labels)

            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images.detach())
            fake_labels = torch.zeros_like(fake_outputs).to(device)
            fake_loss = criterion(fake_outputs, fake_labels)

            disc_loss = disc_weight * (real_loss + fake_loss)
            disc_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            disc_optim.step()

            gen_optim.zero_grad()
            outputs = discriminator(fake_images)
            gen_loss = gen_weight * criterion(outputs, real_labels)

            gen_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            gen_optim.step()

            loop.set_description(
                f'epoch:{epoch}, batch:{i}, disc_loss:{disc_loss.item()}, gen_loss: {gen_loss.item()}')
            loop.update(1)

            disc_losses.append(disc_loss.item())
            gen_losses.append(gen_loss.item())

            # checkpoint every checkpoint_num batches
            if i % checkpoint_num == 0:
                checkpoint = {
                    'epoch': epoch,
                    'batch': i,
                    'discriminator_state_dict': discriminator.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_optimizer_state_dict': disc_optim.state_dict(),
                    'generator_optimizer_state_dict': gen_optim.state_dict(),
                    'discriminator_loss': disc_loss.item(),
                    'generator_loss': gen_loss.item()
                }
                torch.save(
                    checkpoint, f'checkpoints/checkpoint_{epoch}_{i}.pth')

                generate_images(generator, device, images_to_generate, latent_size,
                                image_size, f'generated_images/generated_after_{epoch}_{i}.png')

    loop.close()

    # save the final checkpoint
    checkpoint = {
        'epoch': epoch,
        'batch': i,
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_optimizer_state_dict': disc_optim.state_dict(),
        'generator_optimizer_state_dict': gen_optim.state_dict(),
        'discriminator_loss': disc_loss.item(),
        'generator_loss': gen_loss.item()
    }
    torch.save(checkpoint, f'checkpoints/final_checkpoint.pth')

    # save the losses
    torch.save(disc_losses, 'losses/disc_losses.pth')
    torch.save(gen_losses, 'losses/gen_losses.pth')

    # plot the losses
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.plot(gen_losses, label='Generator Loss')
    plt.legend()
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('GAN Losses')
    plt.savefig('losses/losses.png')

    # generate images
    generate_images(generator, device, images_to_generate, latent_size,
                    image_size, 'generated_images/final_images.png')
