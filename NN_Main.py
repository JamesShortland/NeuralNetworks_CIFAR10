import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
print(f"Using device: {device}")  # Verify it says "mps"

train_batch_losses = []
train_epoch_accuracies = []
test_epoch_accuracies = []

best_test_accuracy = 0.0
best_model_state = None

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True,
                                            transform=train_transform)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=64,
                                              shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                          shuffle=True, num_workers=0)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck']


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv, num_kernel_size, num_padding):
        super(Block, self).__init__()

        self.num_conv = num_conv

        for i in range(num_conv):
            conv_layer = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=num_kernel_size,
                                   padding=num_padding)

            # Apply Xavier initialization to the conv layer
            nn.init.kaiming_normal_(conv_layer.weight, mode='fan_in', nonlinearity='relu')
            if conv_layer.bias is not None:
                nn.init.constant_(conv_layer.bias, 0)

            # Create the sequential block
            conv = nn.Sequential(
                conv_layer,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

            setattr(self, f'conv{i + 1}', conv)

        self.fc = nn.Linear(in_channels, num_conv)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        m = x.mean(dim=[2, 3])
        a = self.fc(m)
        a = F.softmax(a, dim=1)

        out = torch.zeros_like(getattr(self, 'conv1')(x))

        for i in range(self.num_conv):
            conv = getattr(self, f'conv{i + 1}')
            out += a[:, i].view(-1, 1, 1, 1) * conv(x)

        out = self.pool(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, block_configs, num_classes=10):
        super(ConvNet, self).__init__()

        self.blocks = nn.ModuleList()

        for i, cfg in enumerate(block_configs):
            in_ch = cfg['in_channels']
            out_ch = cfg['out_channels']
            num_conv = cfg['num_conv']
            num_kernel_size = cfg['num_kernel_size']
            num_padding = cfg['num_padding']

            self.blocks.append(Block(in_ch, out_ch, num_conv, num_kernel_size, num_padding))

        final_out_channels = block_configs[-1]['out_channels']
        self.output_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(final_out_channels, num_classes))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        m = x.mean(dim=[2, 3])
        logits = self.output_layer(m)

        return logits

model = ConvNet(block_configs=[
    {'in_channels': 3, 'out_channels': 64, 'num_conv': 5, 'num_kernel_size': 5, 'num_padding': 2},
    {'in_channels': 64, 'out_channels': 128, 'num_conv': 5, 'num_kernel_size': 3, 'num_padding': 1},
    {'in_channels': 128, 'out_channels': 256, 'num_conv': 5, 'num_kernel_size': 3, 'num_padding': 1},
    {'in_channels': 256, 'out_channels': 512, 'num_conv': 5, 'num_kernel_size': 3, 'num_padding': 1}])

model.to(device)
print(f"Model is actually on: {next(model.parameters()).device}")

epochs = 150
# Loss & optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(training_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_batch_losses.append(loss.item())  # Track batch loss

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    scheduler.step()

    train_acc = correct / total
    train_epoch_accuracies.append(train_acc)  # Track training accuracy

    test_acc = evaluate(model, test_loader, device)  # Test set as validation
    test_epoch_accuracies.append(test_acc)          # Track test accuracy

    if test_acc > best_test_accuracy:
        best_test_accuracy = test_acc
        best_model_state = model.state_dict()
        print('This was the best model yet!')
        # Save best model in memory

    print(f"Epoch {epoch+1} - Loss: {running_loss / len(training_loader):.4f} | "
          f"Train Accuracy: {train_acc*100:.2f}% | Test Accuracy: {test_acc*100:.2f}%")


print(best_model_state)
model.load_state_dict(best_model_state)
final_test_acc = evaluate(model, test_loader, device)
print(f"\nFinal Test Accuracy using best model: {final_test_acc*100:.2f}%")

# Plot training loss per batch
plt.figure(figsize=(10, 4))
plt.plot(train_batch_losses, label='Batch Loss')
plt.title('Training Loss per Batch')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and test accuracy per epoch
plt.figure(figsize=(10, 4))
plt.plot(train_epoch_accuracies, label='Train Accuracy')
plt.plot(test_epoch_accuracies, label='Test Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

