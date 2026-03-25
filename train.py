import torch
from torch import nn
import wandb
from models.custom_model import CustomNet
from dataset.dataloader import get_dataloaders
from eval import validate

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    
    # Log su wandb per il training
    wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})

def main():
    wandb.init(project="faimdl-lab3")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inizio training su: {device}")
    print("-" * 30)

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    train_loader, val_loader = get_dataloaders("tiny-imagenet/tiny-imagenet-200", batch_size=32)
    
    best_acc = 0
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        # Training
        train(epoch, model, train_loader, criterion, optimizer, device)

        # Validazione
        val_accuracy = validate(model, val_loader, criterion, device)
        
        # Log su wandb per la validazione
        wandb.log({"val_accuracy": val_accuracy})

        # Salvataggio del modello migliore
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            import os
            # Crea il percorso su Drive
            save_path = '/content/drive/MyDrive/MLDL_Lab3_Checkpoints'
            os.makedirs(save_path, exist_ok=True)
            # Salva il file dentro Drive
            torch.save(model.state_dict(), os.path.join(save_path, 'best_custom_model.pth'))
            print(f' NEW BEST! Model saved with {best_acc:.2f}% accuracy')

    print("-" * 30)
    print(f'Training completato. Miglior accuratezza: {best_acc:.2f}%')

if __name__ == "__main__":
    main()