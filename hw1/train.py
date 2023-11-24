# import your model from net.py
import torch
import torch.nn as nn   
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils import data
from PIL import Image
import argparse
from tqdm import tqdm
from torchvision import transforms
import os
from net import my_network


'''
    You can add any other package, class and function if you need.
    You should read the .jpg from "./dataset/train/" and save your weight to "./w_{student_id}.pth"
'''
def getData(mode):
    if(mode == "train"):
        df = pd.read_csv('new_train.csv')
        name = df['name'].tolist()
        label = df['label'].tolist()
        #convert origin labels to one hot vactor 
        total_classes = 12
        one_hot_labels = torch.zeros(len(label), total_classes)
        for i, label in enumerate(label):
            one_hot_labels[i, label] = 1

        return name, one_hot_labels
    elif(mode == "valid"):
        df = pd.read_csv("new_valid.csv")
        name = df['name'].tolist()
        label = df['label'].tolist()
        return name, label
    
class Dataset_Sport(data.Dataset):
    def __init__(self,root = "",mode = "train",transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.img_name, self.labels = getData(mode)
        print(f"{self.mode} dataset > Found %d images..."%(len(self.img_name)))

    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.root, os.path.normpath(self.img_name[idx]))
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def save_ckpt(args, model_weight, path, acc):
    torch.save({
        "state_dict": model_weight,
        "acc": acc
    }, os.path.join(args.save_root,path))


def train(args):
    # TODO
    model = my_network() # load model   
    total_params = sum(p.numel() for p in model.parameters())
    print("總參數數量：", total_params)
    best_acc = 0 
    if args.ckpt_path != None:
        if(os.path.exists(args.ckpt_path)):
            checkpoint = torch.load(args.ckpt_path)
            model.load_state_dict(checkpoint['state_dict'], strict=True) 
            best_acc = checkpoint["acc"]
            print("Load weight successful !!!")
        else:
            print("Load weight fail because the path doesn't exist !!!")
    model.to(args.device)

    train_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
        transforms.RandomRotation(30),  # 隨機旋轉圖像
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3,sigma=0.1)]),  # 添加高斯模糊，以20%的機率進行處理
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 添加色彩增強，以50%的機率進行處理

        #transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.6, 1.5), shear=10)], p=0.7), 

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.2229, 0.2224, 0.225])
    ])
    val_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.2229, 0.2224, 0.225])
    ])

    dataset = Dataset_Sport(root = args.dataset_root, mode ="train", transform= train_data_transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers= args.num_workers, shuffle=True)
    dataset = Dataset_Sport(root = args.dataset_root, mode ="valid", transform= val_data_transform)
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers= args.num_workers, shuffle=True)

    print("Start Training!!!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)

    for i in range(args.num_epoch):
        model.train()
        total_loss = 0
        batch_iterator = tqdm(enumerate(train_loader, 0), total=len(train_loader))
        for batch_idx, (inputs, labels) in batch_iterator: 
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
            batch_iterator.set_description(f'Lr: {scheduler.get_last_lr()[0]} Batch Loss: {loss.item():.4f}')
        scheduler.step()
        print(f"Epoch [{i}] Training AVG loss is {total_loss/(batch_idx+1)}")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model.forward(inputs)
                predicted = torch.argmax(outputs, dim=1)
                #print(f"The predicted is {predicted}")
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch [{i}] Testing Accuracy: {accuracy:.2f}%')

        if(accuracy>best_acc):
            best_acc = accuracy
            save_ckpt(args,model.state_dict(),"w_312581006.pth", best_acc)
            print("\033[101;97m" +f"Nice !!! Save the best ckpt to {args.save_root}/w_312581006" + "\033[0m")
            continue
        if(i % args.per_save == 0):
            save_ckpt(args,model.state_dict(),f"epoch_{i}.pth",accuracy)
            print("\033[47;30m"+f"Save ckpt to {args.save_root}/epoch_{i}"+ "\033[0m")

    print("Training is over!!!")

def main(args):
    args.need_update_lr = False
    if args.device == "cuda" and torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))  # 显示GPU的名称
    else:
        args.device = "cpu"
        print("Using CPU.")
    train(args)

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=8)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--dataset_root',  type=str, default="dataset/train",  help="The path to your dataset")
    parser.add_argument('--save_root',     type=str, default="ckpt",  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=60,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=8,      help="Save checkpoint every seted epoch")
    parser.add_argument('--ckpt_path',     type=str,   default=None, help="The path of your checkpoints")  

    args = parser.parse_args()
    main(args)