import torch
from torch import nn
from models.custom_model import CustomNet
from dataset.dataloader import get_dataloaders

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomNet().to(device)
    
    # Carica i pesi del modello addestrato
    model.load_state_dict(torch.load("best_custom_model.pth"))
    
    criterion = nn.CrossEntropyLoss()
    _, val_loader = get_dataloaders("tiny-imagenet/tiny-imagenet-200", batch_size=32)
    
    validate(model, val_loader, criterion, device)