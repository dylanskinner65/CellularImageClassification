from torch.utils.data import DataLoader, dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import image_dataset
import torch.optim as optim
from torchvision import transforms, models
from torchvision.utils import save_image
from tqdm import tqdm

# define the generator network


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.fc2(self.relu(self.fc1(x))))

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


if __name__ == '__main__':
    # set manual seed
    torch.manual_seed(42)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_size = 100
    hidden_size = 256
    image_size = 512
    lr = .00001
    epochs = 1
    batch_size = 32
    images_to_generate = 24

    dataset = image_dataset.ImageDataset(transform=transforms.ToTensor())

    generator = Generator(latent_size, hidden_size,
                          image_size * image_size).to(device)
    discrimator = Discriminator(
        image_size * image_size, hidden_size, 1).to(device)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    criterion = nn.BCELoss()
    gen_optim = optim.Adam(generator.parameters(), lr)
    disc_optim = optim.Adam(discrimator.parameters(), lr)

    loop = tqdm(total=len(dataloader), position=0, leave=False)
    for epoch in range(epochs):
        for i, (real_image, _) in enumerate(dataloader):

            real_image = real_image.view(-1,
                                         image_size * image_size).to(device)

            noise = torch.randn(batch_size, latent_size).to(device)

            # train the discriminator
            disc_optim.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)

            real_outputs = discrimator(real_image)
            real_loss = criterion(real_outputs, real_labels)

            fake_images = generator(noise)
            fake_outputs = discrimator(fake_images.detach())
            fake_labels = torch.zeros_like(fake_outputs).to(device)
            fake_loss = criterion(fake_outputs, fake_labels)

            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_optim.step()

            gen_optim.zero_grad()
            outputs = discrimator(fake_images)
            gen_loss = criterion(outputs, real_labels)

            gen_loss.backward()
            gen_optim.step()

            loop.set_description(
                f'epoch:{epoch}, batch:{i}, disc_loss:{disc_loss.item()}, gen_loss: {gen_loss.item()}')
            loop.update(1)

    loop.close()

    with torch.no_grad():
        noise = torch.randn(images_to_generate, latent_size)
        generated_images = generator(noise).view(-1, 1, image_size, image_size)
        save_image(generated_images, "generated_images.png")
