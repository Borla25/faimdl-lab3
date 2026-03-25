import numpy as np
import matplotlib.pyplot as plt

def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

def visualize_classes(dataloader_train):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten() 
    classes_sampled = []
    found_classes = 0

    for inputs, labels in dataloader_train:
        for j in range(len(labels)):
            label_corrente = labels[j].item()

            if label_corrente not in classes_sampled and found_classes < 10:
                classes_sampled.append(label_corrente)
                img = denormalize(inputs[j])
                axes[found_classes].imshow(img)
                found_classes += 1
        if found_classes >= 10:
            break
    plt.show()