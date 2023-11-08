"""
Vit for CIFAR10 classification task
Hsiu's practice version
"""
import os
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.cifar import CIFAR10

np.random.seed(0)
torch.manual_seed(0)

def patchify(images, n_patches):
    n, c, h, w = images.shape

    patches = torch.zeros(n, n_patches**2, h*w*c//n_patches**2)
    patch_size = h//n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patches[idx, i*n_patches+j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i/(10000**(j/d))) if j%2==0 else np.cos(i/(10000**((j-1)/d)))
    return result

class MyMSA(nn.Module):
    def __init__(self, d, n_heads):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        d_head = int(d/n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head*self.d_head : (head+1)*self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q@k.T/(self.d_head**0.5))
                seq_result.append(attention@v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio*hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio*hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x+self.mhsa(self.norm1(x))
        out = out+self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        #Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.patch_size = (chw[1]/n_patches, chw[2]/n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0]*self.patch_size[0]*self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        #tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))]) 

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classificaiton token only
        out = out[:, 0]

        return self.mlp(out)    # Map to output dimension, output category distribution
    

# Main function
# adaptable parameters
# 1. depends on dataset
CHW = (3, 32, 32)
N_PATCHES = 8
OUT_D = 10

# 2. depends on architecture
N_BLOCKS = 6
HIDDEN_D = 16
N_HEADS = 4

# 3. depends on training
BATCH_SIZE = 32
EPOCHS = 48 
LR = 0.001

# Loading data
train = CIFAR10(root='./cifar10', train=True, transform=ToTensor(), download=True)
test = CIFAR10(root='./cifar10', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(train, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test, shuffle=False, batch_size=BATCH_SIZE)

# Create model and define training options
model = MyViT(chw=CHW, n_patches=N_PATCHES, n_blocks=N_BLOCKS, hidden_d=HIDDEN_D, n_heads=N_HEADS, out_d=OUT_D)

# Training loop
optimizer = Adam(model.parameters(), lr=LR)
loss_func = CrossEntropyLoss()

if os.path.exists("Cifar10.h5"):
    model.load_state_dict(torch.load("Cifar10.h5"))

for epoch in range(EPOCHS):
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        y_predict = model(x)
        loss = loss_func(y_predict, y)
        train_loss += loss.detach().item()/len(test_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch %d/%d loss: %.2f" % (epoch+1, EPOCHS, loss))

    if epoch%20 == 0: 
        torch.save(model.state_dict(), "Cifar10.h5")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            y_predict = model(x)
            loss = loss_func(y_predict, y)
            test_loss += loss.detach().item()/len(test_loader)
    
            correct += torch.sum(torch.argmax(y_predict, dim=1) == y).detach().item()
            total += len(x)
        print("Test loss: %.2f" % (test_loss))
        print("Test accuracy: %.2f%%" % (correct/total*100))
    
        with open("cifar10_accuracy.txt", "a") as file:
            file.write(f"Epoch {epoch+3}: Test loss = {test_loss:.2f}, Test accuracy = {correct / total * 100:.2f}%\n")
