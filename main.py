import torch
import argparse
import datetime

import numpy as np
from tqdm import tqdm
from dataset import Dataset
from trainer import valid, train
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForMultipleChoice, AutoTokenizer, AdamW


now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
writer = SummaryWriter()
parser = argparse.ArgumentParser()

parser.add_argument('--patience' ,  type = int, default=3)
parser.add_argument('--batch_size' , type = int, default=8)
parser.add_argument('--max_epoch' ,  type = int, default=20)
parser.add_argument('--base_trained_model', type = str, default = 'bert-base-uncased', help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--gpu_number' , type = int,  default = 0, help = 'which GPU will you use?')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--log_file' , type = str,  default = f'logs/log_{now_time}.txt', help = 'Is this debuggin mode?')
parser.add_argument('--dataset_name' , type = str,  default = 'race', help = 'race')
parser.add_argument('--dataset_option' , type = str,  default = 'middle', help = 'all|middle|high')
parser.add_argument('--max_length' , type = int,  default = 128, help = 'max length')
parser.add_argument('--max_options' , type = int,  default = 4, help = 'max number of options')




args = parser.parse_args()

if __name__ =="__main__":

    tokenizer = AutoTokenizer.from_pretrained(args.base_trained_model, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(args.base_trained_model)
    train_dataset = Dataset(args.dataset_name, args.dataset_option, tokenizer, args.max_length, args.max_options, "train")
    val_dataset = Dataset(args.dataset_name, args.dataset_option, tokenizer,  args.max_length, args.max_options,"validation") 

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    dev_loader = DataLoader(val_dataset, args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    log_file = open(args.log_file, 'w')

    device = torch.device(f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU


    if args.pretrained_model:
        print("use trained model")
        log_file.write("use trained model")
        model.load_state_dict(torch.load(args.pretrained_model))

    model.to(device)
    penalty = 0
    min_loss = float('inf')

    for epoch in range(args.max_epoch):
        print(f"Epoch : {epoch}")
        train(model, train_loader, optimizer, device)
        ACC = 0
        anss, preds, loss = valid(model, dev_loader, device, tokenizer,log_file)
        for iter, (ans, pred) in enumerate(zip(anss, preds)):
            ACC += accuracy_score(ans, pred)

        print("Epoch : %d, ACC : %.04f, Loss : %.04f" % (epoch, ACC/iter, loss))
        log_file.writelines("Epoch : %d, ACC : %.04f, Loss : %.04f" % (epoch, ACC/iter, loss))

        writer.add_scalar("ACC", ACC/iter, epoch)
        writer.add_scalar("loss",loss, epoch)


        if loss < min_loss:
            print("New best")
            min_loss = loss
            penalty = 0
            if not args.debugging:
                torch.save(model.state_dict(), f"model/{args.dataset_name}.pt")
        else:
            penalty +=1
            if penalty>args.patience:
                print(f"early stopping at epoch {epoch}")
                break
        writer.close()
        log_file.close()


