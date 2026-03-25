import os
import urllib.request
import zipfile
import shutil

dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
zip_path = "tiny-imagenet-200.zip"
extract_dir = "tiny-imagenet"

if not os.path.exists(extract_dir):
    print("Download di Tiny-ImageNet in corso...")
    urllib.request.urlretrieve(dataset_url, zip_path)
    print("Estrazione in corso...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Riorganizzazione della cartella di validazione
    val_dir = os.path.join(extract_dir, 'tiny-imagenet-200', 'val')
    with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
        val_dict = {line.split('\t')[0]: line.split('\t')[1] for line in f.readlines()}
    
    for img, folder in val_dict.items():
        newpath = os.path.join(val_dir, folder)
        os.makedirs(newpath, exist_ok=True)
        img_path = os.path.join(val_dir, 'images', img)
        if os.path.exists(img_path):
            shutil.move(img_path, os.path.join(newpath, img))
            
    shutil.rmtree(os.path.join(val_dir, 'images'))
    print("Dataset preparato con successo.")
else:
    print("Il dataset esiste già.")