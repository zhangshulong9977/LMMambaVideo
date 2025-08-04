import cv2
import torch
import pandas as pd
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import random
import os
import numpy as np
from model import LHMambaVideo

class VideoData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir  
        self.data = pd.read_csv(self.data_dir) 
        self.video = self.data.iloc[:, 0]
        self.video_list = self.video.values.tolist()
        self.label = self.data.iloc[:, 1]
        self.label_list = self.label.values.tolist()

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        frames = []
        while frame_count <= total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            image_tensor = transform(frame_pil)
            frames.append(image_tensor)
            frame_count += 1
        cap.release()
        video_tensor = torch.stack(frames, dim=0)
        video_tensor = video_tensor.permute(1, 0, 2, 3) 
    
        label = self.label_list[idx]
        return video_tensor, label
    
    def __len__(self):
        return len(self.data)

def train(device, net, train_loader, val_loader, loss_fn, optimizer, epochs, path_save_model):
    device = device[0] if isinstance(device, list) else device
    net = net.to(device)
    best_acc = 0
    current_patience = 0
    for epoch in range(epochs):
        train_loss, train_acc = 0, 0
        net.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("{}loss{}".format(i+1, loss))
            train_loss += loss.item() 
            acc = (outputs.argmax(1) == y).sum().item() 
            train_acc += acc 
        avg_train_loss = train_loss/(i+1)
        avg_train_acc = train_acc/len(train_loader.dataset)
        print("train_loss:", avg_train_loss)
        print("train_acc:", avg_train_acc)

        val_loss, val_acc = 0, 0
        net.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                loss = loss_fn(outputs, y)

                val_loss += loss.item()
                acc = (outputs.argmax(1) == y).sum().item()
                val_acc += acc
        avg_val_loss = val_loss/(i+1)
        avg_val_acc = val_acc/len(val_loader.dataset)
        print("val_loss:", avg_val_loss)
        print("val_acc:", avg_val_acc)
        print("Current LR:", optimizer.param_groups[0]['lr'])
        if best_acc < avg_val_acc:
            print(f'val_acc creased ({best_acc:.6f} --> {avg_val_acc:.6f})')
            best_acc = avg_val_acc
            current_patience = 0
        else:
            current_patience += 1
            print(f'val_acc not creased, best val_acc:{best_acc:.6f}, current_patience:{current_patience}')
            if current_patience >= 10:
                for group in optimizer.param_groups:
                    group['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                current_patience = 0
    torch.save(net.state_dict(), path_save_model)

def set_seed(seed):
    random.seed(seed)                       
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)                   
    torch.manual_seed(seed)                 
    torch.cuda.manual_seed(seed)            
    torch.cuda.manual_seed_all(seed)        
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False   

if __name__ == "__main__":
    set_seed(77)
    train_csv = "yourdata/train.csv"
    val_csv = "yourdata/val.csv"
    train_set = VideoData(train_csv)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)
    val_set = VideoData(val_csv) 
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)
    dim = 96
    depths = [1, 3, 8]
    APConv = True
    LHMamba = True
    num_frames = train_set[0][0].shape[1]
    model = LHMambaVideo(dim = dim, depths=depths, num_classes=4, apconv=APConv, lhmamba=LHMamba, num_frames=num_frames)
    path_save_model = "yourmodel.pth"
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
    epochs = 200
    device = [torch.device(f'cuda:{i}') for i in range(2)]
    model = nn.DataParallel(model, device_ids=device)
    train(device, model, train_loader, val_loader,loss_fn, optimizer, epochs,path_save_model)
