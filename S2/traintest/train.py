import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

train_losses = []
train_acc = []


def train(model, device, trainloader, epoch):
    running_loss = 0.00
    correct = 0.0
    processed = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    pbar = tqdm(trainloader)
    for i, data in enumerate(pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        output = outputs.argmax(dim=1, keepdim=True)
        correct = correct + output.eq(labels.view_as(output)).sum().item()
        processed = processed + len(inputs)
        pbar.set_description(desc= f'Loss={loss.item()} Accuracy={100*correct/processed:.2f}')
        train_acc.append(100*correct/processed)
