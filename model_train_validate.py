from torchvision import models
import torch.optim as optim
import torch.nn as nn
from prepare_data import prep
from generate_model import *
from test_acc import validate
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

pretrained = False
model_name = 'mobile_net'
if model_name == 'mobile_net':
    model = gen_mobile_net(pretrained).to(device)
elif model_name == 'squeeze_net':
    model = gen_squeeze_net(pretrained).to(device)

#setting the hyperparameters
batch_size = 64
epochs = 10
lr = 8e-5

train_loader, val_loader, test_loader = prep(batch_size= batch_size, shuffle= True)

print("training images: ", len(train_loader)*batch_size)
print("validation images: ", len(val_loader))

#setting the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_used = 'adam'
optimizer = optim.Adam(model.parameters(), lr, weight_decay= 5e-4)

#setting up path for saving model weights
directory = rf'.\{model_name}_saves'

if not os.path.exists(directory):
    os.makedirs(directory)

ls = os.listdir(directory)
if len(ls) == 0:
    new_save = '1'
else:
    ls.sort()
    new_save = str(int(ls[-1]) + 1)

save_dir = os.path.join(directory, new_save)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

best_model = os.path.join(save_dir,'best_model.pth')
latest_model = os.path.join(save_dir,'latest_model.pth')


#doing the training and validation
best_loss = 100
train_loss = []
val_loss = []

for epoch in range(epochs):
    model.train()
    avg__train_loss = 0;
    avg_val_loss = 0
    for idx, data in enumerate(train_loader):
        batch, labels = data
        batch = batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        avg__train_loss += loss.item()
        if(idx%10 == 0):
            print(f"step {idx + 1}/{len(train_loader)}: ", loss.item())
    
    avg__train_loss /= len(train_loader)
    print(f"epoch {epoch + 1} loss: ", avg__train_loss)

    train_loss.append(avg__train_loss)

    model.eval()
    for idx, data in enumerate(val_loader):
        batch, labels = data
        batch = batch.to(device)
        labels = labels.to(device)

        outputs = model(batch)
        loss = criterion(outputs, labels)
        avg_val_loss += loss.item()

        if(idx % 2000 == 0):
            print(f"step {idx + 1}/{len(val_loader)}: ", loss.item())
    
    avg_val_loss /= len(val_loader)
    print(f"average validation loss: ", avg_val_loss)

    if avg_val_loss < best_loss:
            print("saving best model")
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model)

    val_loss.append(avg_val_loss)

    if epoch == epochs - 1:
        torch.save(model.state_dict(), latest_model)

print(best_loss)

fig = plt.figure()
plt.plot([i for i in range(epochs)], train_loss, label = 'train loss')
plt.plot([i for i in range(epochs)], val_loss, label = 'val loss')
plt.legend()
fig.savefig(os.path.join(save_dir, 'plot.jpg'))

if model_name == 'mobile_net':
    model = gen_mobile_net(False).to(device)
elif model_name == 'squeeze_net':
    model = gen_squeeze_net(False).to(device)
model.load_state_dict(torch.load(best_model))

val_acc, confusion = validate(model, device)

confusion.savefig(os.path.join(save_dir, 'acc.jpg'))

text = open(os.path.join(save_dir, 'summary.txt'), mode = "w+")
text.write(f'model: {model_name}\n')
text.write(f'pretrained: {pretrained}\n')
text.write(f'learning rate: {lr}\n')
text.write(f'epcohs_trained: {epochs}\n')
text.write(f'batch size: {batch_size}\n')
text.write(f'optimizer: {optimizer_used}\n')
text.write(f'best loss: {best_loss}\n')
text.write(f'best test acc: {val_acc}')
text.close()