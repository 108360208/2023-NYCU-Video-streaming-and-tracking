# import your model from net.py
import torch
import torch.nn as nn   
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils import data
import glob
from PIL import Image
import argparse
from tqdm import tqdm
from torchvision import transforms
import os
import re

from net import my_network
'''
    You can add any other package, class and function if you need.
    You should read the .jpg files located in "./dataset/test/", make predictions based on the weight file "./w_{student_id}.pth", and save the results to "./pred_{student_id}.csv".
'''
class Dataset_Sport(data.Dataset):
    def __init__(self,root = "", transform=None):
        self.root = root
        self.transform = transform
        self.img_name_list = glob.glob(os.path.join(root, '*.jpg'))
        
        print(f"Test dataset > Found %d images..."%(len(self.img_name_list)))

    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self,idx):
       
        image = Image.open(self.img_name_list[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
def test(args):
    model = my_network() # load model   
    total_params = sum(p.numel() for p in model.parameters())
    print("總參數數量：", total_params)
    if args.ckpt_path != None:
        if(os.path.exists(args.ckpt_path)):
            if args.device == "cuda":
                checkpoint = torch.load(args.ckpt_path)
                model.load_state_dict(checkpoint['state_dict'], strict=True) 
                print(f"acc: {checkpoint['acc']}")
                print("Load weight successful !!!")
            else:
                checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint['state_dict'], strict=True) 
                print("Load weight successful !!!")
        else:
            print("Load weight fail because the path doesn't exist !!!")
            return
    model.to(args.device)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4782, 0.4786, 0.4373],
                            std=[0.2747, 0.2645, 0.2812])
    ])
    dataset = Dataset_Sport(root = args.dataset_root, transform = data_transform)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers= args.num_workers, shuffle=False)
    print("Start Testing!!!")
    model.eval()
    with torch.no_grad():
        all_predictions = []
        for inputs in tqdm(test_loader, total=len(test_loader)):
            inputs = inputs.to(args.device)
            outputs = model.forward(inputs)
            predicted = torch.argmax(outputs, dim=1)
            all_predictions.extend(predicted.cpu().numpy())  

    file_names = [os.path.basename(file) for file in dataset.img_name_list]
    data = list(zip(file_names, all_predictions))

    sorted_data = sorted(data, key=lambda x: int(re.search(r'(\d+).jpg', x[0]).group(1)))

    df = pd.DataFrame(sorted_data, columns=['name', 'label'])

    output_csv = 'pred_312581006.csv'
    df.to_csv(os.path.join(args.save_root,output_csv), index=False)
    print(f"Testing finish!!! The result is be saved at {output_csv}.")
    
def main(args):
    if args.device == "cuda" and torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))  # 显示GPU的名称
    else:
        args.device = "cpu"
        print("Using CPU.")
    test(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=8)
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--dataset_root',  type=str, default="dataset/test",  help="The path to your dataset")
    parser.add_argument('--save_root',     type=str, default="",  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--ckpt_path',     type=str, default="w_312581006.pth", help="The path of your checkpoints")  
    args = parser.parse_args()

    main(args)