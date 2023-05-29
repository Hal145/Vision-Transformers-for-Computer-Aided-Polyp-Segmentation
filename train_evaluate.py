import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models import VisionTransformer
from dataset import PolypDataset



# Set the paths to your dataset images and masks
image_paths = '/mnt/Depolama/BMCOURSES/4th_term/BM686/assignments/Final/Kvasir-SEG/train/images'  # List of image file paths
mask_paths = '/mnt/Depolama/BMCOURSES/4th_term/BM686/assignments/Final/Kvasir-SEG/train/masks'  # List of mask file paths

# Split the dataset into train, validation, and test sets
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
total_samples = len(image_paths)
train_samples = int(train_ratio * total_samples)
val_samples = int(val_ratio * total_samples)
test_samples = int(test_ratio * total_samples)

train_image_paths = image_paths[:train_samples]
train_mask_paths = mask_paths[:train_samples]
val_image_paths = image_paths[train_samples:train_samples + val_samples]
val_mask_paths = mask_paths[train_samples:train_samples + val_samples]
test_image_paths = image_paths[train_samples + val_samples:]
test_mask_paths = mask_paths[train_samples + val_samples:]

# Define the transformations to apply to the images and masks
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other transformations you may need
])

# Create the dataset and data loaders
train_dataset = PolypDataset(train_image_paths, train_mask_paths, transform=transform)
val_dataset = PolypDataset(val_image_paths, val_mask_paths, transform=transform)
test_dataset = PolypDataset(test_image_paths, test_mask_paths, transform=transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss


def test(model, test_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions.append(outputs.detach().cpu())

    return predictions


# Set the hyperparameters for training
num_epochs = 10
learning_rate = 0.001

# Set the device to use (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the vision transformers model
model = VisionTransformer()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move the model to the device
model = model.to(device)

# Training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

    val_loss = evaluate(model, val_loader, criterion, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'polyp_segmentation_model.pt')

# Load the saved model
model = VisionTransformer()
model.load_state_dict(torch.load('polyp_segmentation_model.pt'))
model = model.to(device)


# Test the model
test_predictions = test(model, test_loader, device)

# Perform further processing or evaluation with the predictions


