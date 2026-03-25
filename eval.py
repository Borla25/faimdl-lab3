import torch
from dataset.dataloader import get_dataloaders
from models.custom_model import CustomNet 
import os

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = 'tiny-imagenet/tiny-imagenet-200'
    print("Caricamento dataloader...")
    _, val_loader = get_dataloaders(data_dir, batch_size=32)
    
    print("Inizializzazione modello...")
    model = CustomNet().to(device)
    
    checkpoint_path = '/content/drive/MyDrive/MLDL_Lab3_Checkpoints/best_custom_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Errore critico: File dei pesi non trovato in {checkpoint_path}")
        return
        
    print("Caricamento pesi da Google Drive...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    correct = 0
    total = 0
    print("Inizio calcolo accuratezza sul Validation Set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuratezza finale: {accuracy:.2f}%")

if __name__ == '__main__':
    evaluate()