"""
Attention Rollout of Vit on MNIST classification task
Reference: 
https://colab.research.google.com/github/mashaan14/VisionTransformer-MNIST/blob/main/VisionTransformer_MNIST.ipynb#scrollTo=_tKasVMrfY3I
"""

import numpy as np
from tqdm import tqdm  
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

# 新增的函式庫
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

#-- Building Model Part --#
def patchify(images, n_patches):
    n, c, h, w = images.shape

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches
    
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        d_head = int(d / n_heads)
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

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out    

class ViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(ViT, self).__init__()
            
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
            
        # Input and patches sizes
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

         # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
            
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
            
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
            
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
            
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
            
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
            
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
            
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
            
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
                
        # Getting the classification token only
        out = out[:, 0]
            
        return self.mlp(out) # Map to output dimension, output category distribution  


#-- Training & Testing Part --#
CHW = (1, 28, 28)
N_PATCHES = 7
OUT_D = 10

N_BLOCKS = 4
HIDDEN_D = 8
N_HEADS = 2

BATCH_SIZE = 128
EPOCHS = 5
LR = 0.005

# Loading data
train_data = MNIST(root='./mnist', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='./mnist', train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

# Defining model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(chw=CHW, n_patches=N_PATCHES, n_blocks=N_BLOCKS, hidden_d=HIDDEN_D, n_heads=N_HEADS, out_d=OUT_D)

# Training loop
optimizer = Adam(model.parameters(), lr=LR)
loss_func = CrossEntropyLoss()

for epoch in range(EPOCHS):
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        y_hat = model(x)
        loss = loss_func(y_hat, y)

        train_loss += loss.detach().item() / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch %d/%d loss: %.2f" % (epoch+1, EPOCHS, train_loss))

# Test loop
with torch.no_grad():
    correct, total = 0, 0
    test_loss = 0.0
    for batch in tqdm(test_loader, desc="Testing"):
        x, y = batch
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        test_loss += loss.detach().item() / len(test_loader)

        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().item()
        total += len(x)
    print("Test loss: %.2f" % (test_loss))
    print("Test accuracy: %.2f%%" % (correct / total * 100))  


#-- Visualization Part --#
# Prepare the test sample,
# the process is similar to what ViT does
img_tensor = test_data.data[58].to(device)
patches = patchify(img_tensor.unsqueeze(0).unsqueeze(0), N_PATCHES).to(model.positional_embeddings.device)
tokens = model.linear_mapper(patches.float())
tokens = torch.cat((model.class_token.expand(1, 1, -1), tokens), dim=1)
transformer_input = tokens + model.positional_embeddings.repeat(1, 1, 1)

out = transformer_input.clone()
for block in model.blocks:
    out = block(out)
#transformer_output = out[:, 0]
print("Input tensor to Transformer: ", transformer_input.shape)
#print("Output vector from Transformer:", transformer_output.shape)

# Expand the dimension to hidden_dim*mlp_ratio
mlp_linear_layer = model.blocks[0].mlp[0]
transformer_input = mlp_linear_layer(transformer_input)

# Split qkv into mulitple q, k, and v vectors for multi-head attantion
qkv = transformer_input.reshape(50, 2, 4, 4)
q = qkv[:, 0].permute(1, 0, 2)
k = qkv[:, 1].permute(1, 0, 2)
kT = k.permute(0, 2, 1)
# Attention Matrix
attention_matrix = q @ kT
# Average the attention weights across all heads.
attention_matrix = torch.mean(attention_matrix, dim=0)

# 1) Residual Connection: 
# Add an identity matrix and Renormalize the weights.
I = torch.eye(attention_matrix.size(1)).to(device)
attention_matrix = attention_matrix + I
attention_matrix = attention_matrix / attention_matrix.sum(dim=-1).unsqueeze(-1)

# 2) Linear Combination
# Recursively multiply the weight matrices
joint_attentions = torch.zeros(attention_matrix.size()).to(device)

joint_attentions[0] = attention_matrix[0]
for n in range(1, attention_matrix.size(0)):
    joint_attentions[n] = torch.matmul(attention_matrix[n], joint_attentions[n-1])

attn_heatmap = joint_attentions[0, 1:].reshape((7, 7))
# Use bilinear interpolation to enlarge the heatmap size
attn_heatmap_resized = F.interpolate(attn_heatmap.unsqueeze(0).unsqueeze(0), [28, 28], mode='bilinear').view(28, 28, 1)

# Visualize attention map
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
img = np.asarray(img_tensor.cpu())
ax1.imshow(img, cmap='gray')
ax1.set_title('MNIST test sample')
ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False, labelleft=False, labelright=False)

ax2.imshow(attn_heatmap_resized.detach().cpu().numpy())
ax2.set_title('Attention map')
ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
plt.show()
