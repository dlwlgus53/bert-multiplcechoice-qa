def main():
    parser = argparse.ArgumentParser()
    """몇 개의 node를 사용 할 건가 정의"""
    parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
    """몇 개의 GPU를 각 node에 사용할 건지 정의"""
    parser.add_argument('-g', '--gpus', default=1, type=int,help='number of gpus per node')
    """모든 node 내에서 현재 node의 rank (args.nodes -1)"""
    parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')
    """에폭 수 정의"""
    parser.add_argument('--epochs', default=2, type=int, metavar='N',help='number of total epochs to run')
    args = parser.parse_args()
        
    """총 프로세스를 정의"""
    args.world_size = args.gpus * args.nodes 
    """os.environ['Master_ADDR']과 ['MASTER_PORT']를 통해 멀티프로세싱 모듈이 0번째 프로세스를 바라봄"""
    os.environ['MASTER_ADDR'] = '10.57.23.164'              
    os.environ['MASTER_PORT'] = '8888'    
    """학습 함수를 스레드를 통해 실행"""                  
    mp.spawn(train, nprocs=args.gpus, args=(args,)) 
    
def train(gpu, args):
    """시드 설정"""
    torch.manual_seed(0)
    """모델 정의"""
    model = ConvNet()
    """GPU 설정"""
    torch.cuda.set_device(gpu)
    """GPU에 모델 로드"""
    model.cuda(gpu)
    """배치사이즈 설정"""
    batch_size = 100
    """손실함수 정의"""
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    """옵티마이저 정의"""
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    """토치비전에서 제공하는 데이터셋 로드"""
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    """데이터 로더 설정"""
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
		"""학습 측정 시작"""
    start = datetime.now()
    """한 epoch당 총 학습량"""
    total_step = len(train_loader)
    """학습 시작"""
    for epoch in range(args.epochs):
        """데이터 로더에서 데이터 로드"""
        for i, (images, labels) in enumerate(train_loader):
            """학습 데이터 GPU에 로드"""
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            """Forward"""
            outputs = model(images)
            """손실 값 계산"""
            loss = criterion(outputs, labels)

            """옵티마이저 리셋"""
            optimizer.zero_grad()
            """Backward"""
            loss.backward()
            """옵티마이저 스텝 (가중치 업데이트)"""
            optimizer.step()
            """학습 상태 프린트"""
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
		"""에폭 당 학습시간 프린트 """
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))