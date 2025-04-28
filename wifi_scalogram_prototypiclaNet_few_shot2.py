"""
Prototypical Networks for WiFi Device Identification
Modified from: https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my-first-few-shot-classifier.ipynb

Original Authors:
- Thomas Chaton (Sicara)
- Clément Chadebec (Sicara)
- Lilian Boulard (Sicara)

Modifications by:
- Mutala Mohammed
- at Shanghai Jiao Tong University
    + Added custom WiFi scalogram dataset support
    + Modified architecture for WiFi device identification
    + Adjusted training parameters for smaller dataset
    + Added Windows compatibility fixes
"""

import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from PIL import Image
from tqdm import tqdm
from torchsummary import summary
from prettytable import PrettyTable
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average


class WiFiScalogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[target_class]
                    ))
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [label for (_, label) in self.samples]


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, support_images, support_labels, query_images):
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)

        unique_labels = torch.unique(support_labels)
        z_proto = torch.stack([
            z_support[support_labels == label].mean(0)
            for label in unique_labels
        ])

        return -torch.cdist(z_query, z_proto)


def print_model_summary(model, input_size=(3, 128, 128)):
    """Print model architecture summary"""
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE SUMMARY".center(80))
    print("=" * 80)

    # Backbone summary
    print("\nBACKBONE DETAILS:")
    summary(model.backbone, input_size=input_size, device='cuda')

    # Parameter count
    param_table = PrettyTable(["Layer", "Parameters"])
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            param_table.add_row([name, f"{param_count:,}"])
            total_params += param_count
    print(f"\nTOTAL TRAINABLE PARAMETERS: {total_params:,}")
    print("=" * 80 + "\n")


def evaluate(model, data_loader):
    """Evaluate model performance"""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            support, s_labels, query, q_labels, _ = batch
            scores = model(
                support.cuda(),
                s_labels.cuda(),
                query.cuda()
            )
            preds = scores.argmax(dim=1)
            correct += (preds == q_labels.cuda()).sum().item()
            total += len(q_labels)

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def main():
    # Configuration
    config = {
        "n_way": 4,
        "n_shot": 5,
        "n_query": 15,
        "train_episodes": 10000,
        "test_episodes": 500,
        "learning_rate": 0.001,
        "num_epochs": 10
    }

    # Initialize model
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Flatten()
    model = PrototypicalNetworks(backbone).cuda()
    print_model_summary(model)

    # Dataset paths
    train_path = r"C:\Users\user\PycharmProjects\PyTorch_Project\datasets\WiFi_data_prototype\background"
    test_path = r"C:\Users\user\PycharmProjects\PyTorch_Project\datasets\WiFi_data_prototype\evaluation"

    # Verify paths exist
    print(f"\nTrain path exists: {os.path.exists(train_path)}")
    print(f"Test path exists: {os.path.exists(test_path)}")

    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_set = WiFiScalogramDataset(train_path, train_transform)
    test_set = WiFiScalogramDataset(test_path, test_transform)

    # Print dataset info
    print(f"\nTrain samples: {len(train_set)}")
    print(f"Test samples: {len(test_set)}")
    print(f"Classes: {train_set.classes}")

    # Create data loaders
    train_sampler = TaskSampler(
        train_set,
        n_way=config["n_way"],
        n_shot=config["n_shot"],
        n_query=config["n_query"],
        n_tasks=config["train_episodes"]
    )

    test_sampler = TaskSampler(
        test_set,
        n_way=config["n_way"],
        n_shot=config["n_shot"],
        n_query=config["n_query"],
        n_tasks=config["test_episodes"]
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        losses = []

        for episode, (support, s_labels, query, q_labels, _) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        ):
            optimizer.zero_grad()
            scores = model(
                support.cuda(),
                s_labels.cuda(),
                query.cuda()
            )
            loss = criterion(scores, q_labels.cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            if (episode + 1) % 500 == 0:
                avg_loss = sum(losses[-500:]) / 500
                tqdm.write(f"Episode {episode + 1}: Loss={avg_loss:.4f}")

        # End-of-epoch evaluation
        epoch_acc = evaluate(model, test_loader)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved (Acc: {best_acc:.2f}%)")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()