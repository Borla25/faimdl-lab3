import torch
import os
from dataset.dataloader import get_dataloaders
from models.custom_model import CustomNet

def validate(model, val_loader, criterion, device):
    """Funzione chiamata alla fine di ogni epoca da train.py"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_accuracy = 100. * correct / total
    return val_accuracy

def evaluate_standalone():
    """Funzione chiamata quando esegui !python eval.py nello Step 5"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'tiny-imagenet/tiny-imagenet-200'
    
    print("Caricamento dataloader...")
    _, val_loader = get_dataloaders(data_dir, batch_size=32)
    
    print("Inizializzazione modello...")
    model = CustomNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Cerca il file salvato localmente da train.py
    checkpoint_path = 'best_custom_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Errore: File dei pesi non trovato in {checkpoint_path}")
        return
        
    print("Caricamento pesi...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    print("Inizio calcolo accuratezza sul Validation Set...")
    final_accuracy = validate(model, val_loader, criterion, device)
    print(f"Accuratezza finale: {final_accuracy:.2f}%")

if __name__ == '__main__':
    evaluate_standalone()