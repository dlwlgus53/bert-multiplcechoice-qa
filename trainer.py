import torch
from tqdm import tqdm
import gc
import pdb 
from base_logger import logger

from sklearn.metrics import accuracy_score
def train(gpu, model, train_loader, optimizer):
    model.train()
    if gpu==0: logger.info("Train start")
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = {k:v.cuda(non_blocking = True) for k, v in batch.items()}
        outputs = model(input_ids = batch['input_ids'], token_type_ids = batch['token_type_ids'],\
                            attention_mask=batch['attention_mask'], labels = batch['labels'])
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        
        if (iter + 1) % 10 == 0 and gpu==0:
            logger.info('gpu {} step : {}/{} Loss: {:.4f}'.format(
                gpu,
                iter, 
                str(len(train_loader)),
                loss.detach())
            )
            



def valid(gpu, model, dev_loader):
    model.eval()
    loss_sum = 0
    anss = []
    preds = []
    if gpu==0: logger.info("Validation start")
    with torch.no_grad():
        for iter,batch in enumerate(dev_loader):
            batch = {k:v.cuda(non_blocking = True) for k, v in batch.items()} # 동기적으로 작동하도록!
            outputs = model(input_ids = batch['input_ids'], token_type_ids = batch['token_type_ids'],\
                attention_mask=batch['attention_mask'], labels = batch['labels'])

            anss += batch['labels'].to('cpu').tolist()
            preds += torch.max(outputs[1], axis = 1).indices.to('cpu').tolist()
            loss_sum += outputs[0].detach()
            
            if (iter + 1) % 10 == 0 and gpu == 0:
                logger.info('step : {}/{} Loss: {:.4f}'.format(
                iter, 
                str(len(dev_loader)),
                outputs[0].detach()
                ))
                
    return  anss, preds, loss_sum/iter
        
        