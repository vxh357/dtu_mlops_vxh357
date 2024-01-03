import torch

from torch import nn 

batch_size = 256
lr = 1e-4
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, train_labels = [ ], [ ]
for i in range(5):
    train_data.append(torch.load(f"train_images_{i}.pt"))
    train_labels.append(torch.load(f"train_target_{i}.pt"))

train_data = torch.cat(train_data, dim=0)
train_labels = torch.cat(train_labels, dim=0)

test_data = torch.load("test_images.pt")
test_labels = torch.load("test_target.pt")

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

train_data = train_data.unsqueeze(1)
test_data = test_data.unsqueeze(1)

print(train_data.shape)
print(test_data.shape)

train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_data, train_labels),
    batch_size=batch_size,
)
test_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_data, test_labels),
    batch_size=batch_size,
)

model = nn.Sequential(
    nn.Conv2d(1, 32, 3), # [B, 1, 28, 28] -> [B, 32, 26, 26]
    nn.LeakyReLU(),
    nn.Conv2d(32, 64, 3), # [B, 32, 26, 26] -> [B, 64, 24, 24]
    nn.LeakyReLU(),
    nn.MaxPool2d(2), # [B, 64, 24, 24] -> [B, 64, 12, 12]
    nn.Flatten(),    # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
    nn.Linear(64 * 12 *12 * 10), 
)
model = model.to(device)
print(model(test_data[:1].to(device)).shape)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = 

for epoch in range(num_epochs):
    for batch in train_dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)







model.eval()

test_preds = [ ]
test_labels = [ ]
with torch.no_grad():
    for batch in test_dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)