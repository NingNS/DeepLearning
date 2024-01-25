import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from DataTrain import *
from Unet import *

device = torch.device('cuda')
weightPath = 'weight/unet.pth'
dataPath = 'train2'
savePath1 = 'predict'

if __name__ == '__main__':
    loadData = DataLoader(Data(dataPath), batch_size=4, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weightPath):
        net.load_state_dict(torch.load(weightPath))
        print('load_weight！')
    opt = optim.Adam(net.parameters(), lr=0.01)  # 优化器
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    # loss_fun = nn.CrossEntropyLoss()
    # weight=torch.from_numpy(np.array([50.0,])).float()
    epoch = 1
    N = 256 * 256
    while epoch <= 250:
        for i, (image, label) in enumerate(loadData):
            image, label = image.to(device), label.to(device)
            # print(image)
            # print(label)
            out_image = net(image)
            train_loss = -(2.0 / N) * torch.sum(
                50 * label * torch.log(out_image + 1e-10) + (1 - label) * torch.log(1 - out_image + 1e-10))
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            # print(out_image)
            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            _image = image[0]
            _label = label[0]
            _out_image = out_image[0]

            img1 = torch.stack([_label, _out_image], dim=0)
            save_image(img1, f'{savePath1}/{i}.png')
        if epoch % 1 == 0:
            torch.save(net.state_dict(), weightPath)
            print('save******************************************************')
        epoch += 1
