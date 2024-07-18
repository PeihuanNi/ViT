import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR100(root='D:/Dataset/CIFAR-100', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='D:/Dataset/CIFAR-100', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # shape: (batch_size, embed_dim, num_patches**0.5, num_patches**0.5)
        x = x.flatten(2)  # shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # shape: (batch_size, num_patches, embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, n_patches, dim = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(batch_size, n_patches, self.heads, -1).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, n_patches, dim)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(dim, heads, dim_head)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=100, dim=64, depth=6, heads=8, mlp_dim=128, pool='cls', channels=3, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, channels, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = nn.ModuleList([TransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size, n_patches, _ = x.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.dropout(x)

        for transformer in self.transformer:
            x = transformer(x)

        if self.pool == 'cls':
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return self.mlp_head(x)


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = ViT().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(50):  # 训练10个epoch
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每100个批次打印一次损失
            print(f'Epoch [{epoch + 1}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Finished Training')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
