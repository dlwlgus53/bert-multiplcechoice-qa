import torch
from tqdm import tqdm
import pdb 
def train(model, train_loader, optimizer, device):
        model.train()
        loss_sum = 0
        t_train_loader = tqdm(train_loader)
        for batch in t_train_loader:
            optimizer.zero_grad()
            batch = {k:v.to(device)for k, v in batch.items()}
            outputs = model(input_ids = batch['input_ids'], token_type_ids = batch['token_type_ids'],\
                             attention_mask=batch['attention_mask'], labels = batch['labels'])
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            t_train_loader.set_description("Loss %.04f" % (loss))



def valid(model, dev_loader, device, tokenizer, log_file):

    model.eval()
    anss = []
    preds = []
    loss_sum = 0
    print("Validation start")
    with torch.no_grad():
        log_file.write("\n")
        t_dev_loader = tqdm(dev_loader)
        for iter,batch in enumerate(t_dev_loader):
            anss += batch['labels']
            batch = {k:v.to(device)for k, v in batch.items()}
            outputs = model(input_ids = batch['input_ids'], token_type_ids = batch['token_type_ids'], attention_mask=batch['attention_mask'], labels = batch['labels'])
            loss_sum += outputs[0].to('cpu')
            preds += torch.max(outputs[1], axis = 1).indices.to('cpu')
            
            t_dev_loader.set_description("Loss %.04f  | step %d" % (outputs[0].to('cpu'), iter))
            
    return  anss, preds, loss_sum/iter
        
        