from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os

def get_dataloaders(data_dir, batch_size=32):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_path = f'{data_dir}/train'
    val_path = f'{data_dir}/val'
    
    tiny_imagenet_dataset_train = ImageFolder(root=train_path, transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root=val_path, transform=transform)
    
    # Rileva automaticamente il numero di core disponibili (solitamente 2 su Colab T4)
    workers = os.cpu_count()
    
    train_loader = DataLoader(
        tiny_imagenet_dataset_train, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=workers,
        pin_memory=True       # Accelera il trasferimento verso la GPU
    )
    
    val_loader = DataLoader(
        tiny_imagenet_dataset_val, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    
    return train_loader, val_loader