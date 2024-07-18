import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.optim as optim

def Scaled_Dot_Product_Attention(Q, K, V, embed_size):
    attention_score = torch.matmul(Q, K.mT) / (embed_size ** 0.5)
    attention_weight = torch.softmax(attention_score, dim=1)
    attention_matrix = torch.matmul(attention_weight, V)
    return attention_matrix

class EmbeddingLayer(nn.Module):
    def __init__(self, input_size, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.embedding_layer = nn.Linear(self.input_size, self.embed_size, bias=False)
        # print(self.embedding_layer.weight.shape)

    def forward(self, x):
        # print(f'x shape {x.shape}')
        x = self.embedding_layer(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads=3):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        self.values1 = nn.Linear(embed_size, embed_size, bias=False)
        self.queries1 = nn.Linear(embed_size, embed_size, bias=False)
        self.keys1 = nn.Linear(embed_size, embed_size, bias=False)

        self.values2 = nn.Linear(embed_size, embed_size, bias=False)
        self.queries2 = nn.Linear(embed_size, embed_size, bias=False)
        self.keys2 = nn.Linear(embed_size, embed_size, bias=False)

        self.values3 = nn.Linear(embed_size, embed_size, bias=False)
        self.queries3 = nn.Linear(embed_size, embed_size, bias=False)
        self.keys3 = nn.Linear(embed_size, embed_size, bias=False)

        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # print(x.shape)
        Q1 = self.queries1(x)
        Q2 = self.queries2(x)
        Q3 = self.queries3(x)

        K1 = self.keys1(x)
        K2 = self.keys2(x)
        K3 = self.keys3(x)

        V1 = self.values1(x)
        V2 = self.values2(x)
        V3 = self.values3(x)

        attention_matrix_1 = Scaled_Dot_Product_Attention(Q1, K1, V1, self.embed_size)
        attention_matrix_2 = Scaled_Dot_Product_Attention(Q2, K2, V2, self.embed_size)
        attention_matrix_3 = Scaled_Dot_Product_Attention(Q3, K3, V3, self.embed_size)

        
        attention_matrix_mean = (attention_matrix_1 + attention_matrix_2 + attention_matrix_3) / 3
        # print(f'attention_matrix shape: {attention_matrix_mean.shape}')
        # attention_matrix_mean = attention_matrix_mean.view(1, -1)

        output = self.fc(attention_matrix_mean)

        return output
    
class ViT(nn.Module):
    def __init__(self, embed_size, input_size, num_classes, heads=3):
        super(ViT, self).__init__()
        self.embed_size = embed_size
        self.input_size = input_size
        self.heads = heads
        self.embed_layer = EmbeddingLayer(input_size, embed_size)
        self.mul_head_attn1 = MultiHeadSelfAttention(embed_size, heads=heads)
        self.mul_head_attn2 = MultiHeadSelfAttention(embed_size, heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        self.ffn1 = nn.Linear(embed_size, embed_size, bias=False)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(embed_size, embed_size, bias=False)
        self.fc = nn.Linear(embed_size*input_size, num_classes, bias=False)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.embed_layer(x)
        identity = output
        # print(f'after embedding layer: {output.shape}')
        output = self.mul_head_attn1(output)
        output += identity
        output = self.norm(output)
        identity = output
        output = self.mul_head_attn2(output)
        output += identity
        # print(output.shape)
        output = self.norm(output)
        output = self.ffn1(output)
        output = self.relu(output)
        output = self.ffn2(output)
        output = self.flatten(output)
        output = self.fc(output)
        # print('output')
        # print(output.shape)
        output = self.softmax(output)
        # print(output.shape)
        
        return output

embed_size = 2048
num_classes = 10
batch_size = 256
num_epochs = 100
learning_rate = 2e-4
heads = 3

# 数据预处理
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载数据集
train_dataset = CIFAR10(root='D:/Dataset/CIFAR-10', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CIFAR10(root='D:/Dataset/CIFAR-10', transform=transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = ViT(embed_size, heads=heads, input_size=32, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda')
model.to(device)


correct = 0
total = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images = images.view(-1, 32, 32)
        # print('images and labels have place on gpu')
        # print(f'images shape is: {images.shape}')
        # 前向传播
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Accuracy: {100 * correct / total:.2f}%')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# 验证模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')