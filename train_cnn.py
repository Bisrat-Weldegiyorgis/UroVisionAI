import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("dataset/labels.csv")

# Encode labels
le = LabelEncoder()
df["encoded_label"] = le.fit_transform(df["label"])

# Save label mapping
import pickle
with open("dataset/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -----------------------
# Dataset Class
# -----------------------
class UrineDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image"]
        label = self.df.iloc[idx]["encoded_label"]

        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        return image, torch.tensor(label)

dataset = UrineDataset(df)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# -----------------------
# CNN Model
# -----------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------
# Setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(len(le.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# Training loop
# -----------------------
epochs = 15

for epoch in range(epochs):
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# -----------------------
# Save model
# -----------------------
torch.save(model.state_dict(), "dataset/urine_cnn.pth")

print("Training complete! Model saved.")
