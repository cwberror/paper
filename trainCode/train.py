from module.lzgdModule import Lzgd
from module.FCNModule import FCN
from module.unetplusplusModule import NestedUNet
from torch import nn,optim
import torch.nn.functional as F
import time,os,copy
import pandas as pd
from loss.dice import BCE_DICE_Loss
import torch,cv2
import numpy as np
from tqdm import tqdm
import torchvision


def train_model(model,criterion,optimizer,traindataloader,valdataloader,device,num_epochs = 25):
    """
    :param model: 网络模型
    :param criterion: 损失函数
    :param optimizer: 优化函数
    :param traindataloader: 训练的数据集
    :param valdataloader: 验证的数据集
    :param num_epochs: 训练的轮数
    """
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    print('训练开始！！！')
    start_time=time.time()

    for epoch in range(num_epochs):
        print(f"-------------------现在开始第{epoch+1}轮训练-------------------")
        epoch_time=time.time()
        train_loss = 0.0
        train_num = 0
        val_loss = 0.0
        val_num = 0

       ## 每个epoch包括训练和验证阶段
        model.train()  ## 设置模型为训练模式
        for step,(img,label) in enumerate(traindataloader):
            # print(label.shape)
            optimizer.zero_grad()
            img = img.to(device, dtype=torch.float32)
            # label=label.squeeze()

            label = label.to(device, dtype=torch.float32)
            # print(img.shape,label.shape)
            out = model(img)
            out = F.softmax(out,dim=1)
            pre_lab = torch.argmax(out,1) ## 预测的标签
            loss = criterion(out, label.float()) ## 计算损失函数值
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(label)
            train_num += len(label)

            out = np.array(out.data.cpu()[0])[0]
            out=(out * 255).astype(np.uint8)
            # print(type(out),out.dtype)
            cv2.imwrite(os.path.join('see',os.path.basename(traindataloader.dataset.data_list[step]).split('.')[0]+"_"+str(epoch)+".jpg"),out)
            # label=label.to('cpu').numpy()
            # cv2.imwrite(os.path.join('see',os.path.basename(traindataloader.dataset.data_list[step]).split('.')[0]+"_"+str(epoch)+".jpg"),label)


        train_loss_all.append(train_loss / train_num)
        print('{} Train loss: {:.4f}'.format(epoch+1, train_loss_all[-1]))

        ## 计算一个epoch训练后在验证集上的损失
        with torch.no_grad():
            model.eval() ## 设置模型为验证模式
            for step,(img,label) in enumerate(valdataloader):
                img = img.to(device, dtype=torch.float32)
                label=label.squeeze()
                label = label.to(device, dtype=torch.float32)
                out = model(img)
                out = F.softmax(out,dim=1)
                pre_lab = torch.argmax(out,1) ## 预测的标签
                loss = criterion(out, label) ## 计算损失函数值
                val_loss += loss.item() * len(label)
                val_num += len(label)




            ## 计算一个epoch在验证集上的损失和精度
            val_loss_all.append(val_loss / val_num)
            print('{} Val loss: {:.4f}'.format(epoch+1, val_loss_all[-1]))

            ## 保存最好的网络参数
            if val_loss_all[-1] < best_loss:
                best_loss = val_loss_all[-1]
                best_model_wts = copy.deepcopy(model.state_dict())

        ## 每个epoch花费的时间
        time_use = time.time() - epoch_time
        print(f"第{epoch}轮训练{time_use // 60}分{time_use %60}秒")
    train_process = pd.DataFrame(
        data = {"epoch":range(num_epochs),
                "train_loss_all":train_loss_all,
                "val_loss_all":val_loss_all})
    ## 输出最好的模型
    model.load_state_dict(best_model_wts)
    time_use = time.time() - start_time
    print(f"总耗时{time_use // 60}分{time_use %60}秒")
    return model,train_process




def train_1(model,criterion,optimizer,traindataloader,valdataloader,device,num_epochs = 25):
    # 训练
    num_epoch = 2
    # 训练日志保存
    logfile_dir = 'log/'
    # file_train_loss = open('/Users/lyndsey/NewScenery/ml-homework/Tooth-Detection/log/train_loss.txt', 'w')
    # file_train_acc = open('/Users/lyndsey/NewScenery/ml-homework/Tooth-Detection/log/train_acc.txt', 'w')

    # file_val_loss = open('/Users/lyndsey/NewScenery/ml-homework/Tooth-Detection/log/val_loss.txt', 'w')
    # file_val_acc = open('/Users/lyndsey/NewScenery/ml-homework/Tooth-Detection/log/val_acc.txt', 'w')

    acc_best_wts = model.state_dict()
    best_acc = 0
    iter_count = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, 3, 0.1)

    for epoch in range(num_epochs):
        print(f"-------------------现在开始第{epoch + 1}轮训练-------------------")
        train_loss = 0.0
        train_acc = 0.0
        train_correct = 0
        train_total = 0

        val_loss = 0.0
        val_acc = 0.0
        val_correct = 0
        val_total = 0

        scheduler.step()
        for i, sample_batch in enumerate(traindataloader):
            # print(sample_batch)
            inputs = sample_batch[0].to(device)
            labels = sample_batch[1].to(device)
            # print(inputs,labels)

            # 模型设置为train
            model.train()

            # forward
            outputs = model(inputs)

            # print(labels)
            # loss
            loss = criterion(outputs, labels)

            # forward update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            train_total += labels.size(0)

            print('iter:{}'.format(i))

            if i % 10 == 9:
                for sample_batch in valdataloader:
                    inputs = sample_batch[0].to(device)
                    labels = sample_batch[1].to(device)

                    model.eval()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, prediction = torch.max(outputs, 1)
                    val_correct += ((labels == prediction).sum()).item()
                    val_total += inputs.size(0)
                    val_loss += loss.item()

                val_acc = val_correct / val_total
                print('[{},{}] train_loss = {:.5f} train_acc = {:.5f} val_loss = {:.5f} val_acc = {:.5f}'.format(
                    epoch + 1, i + 1, train_loss / 100, train_correct / train_total, val_loss / len(valdataloader),
                    val_correct / val_total))
                if val_acc > best_acc:
                    best_acc = val_acc
                    acc_best_wts = copy.deepcopy(model.state_dict())

                with open(logfile_dir + 'train_loss.txt', 'a') as f:
                    f.write(str(train_loss / 100) + '\n')
                with open(logfile_dir + 'train_acc.txt', 'a') as f:
                    f.write(str(train_correct / train_total) + '\n')
                with open(logfile_dir + 'val_loss.txt', 'a') as f:
                    f.write(str(val_loss / len(valdataloader)) + '\n')
                with open(logfile_dir + 'val_acc.txt', 'a') as f:
                    f.write(str(val_correct / val_total) + '\n')

                iter_count += 200

                train_loss = 0.0
                train_total = 0
                train_correct = 0
                val_correct = 0
                val_total = 0
                val_loss = 0

    print('Train finish!')
    # 保存模型
    model_file = os.getcwd()
    # with open(model_file+'/model_squeezenet_teeth_1.pth','a') as f:
    #     torch.save(acc_best_wts,f)
    torch.save(acc_best_wts, model_file + '/model_squeezenet_utk_face_1.pth')
    print('Model save ok!')



def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def train_fn(loader, model, optimizer, loss_fn, scaler,device):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        # targets = targets.float().unsqueeze(1).to(device=device)
        targets = targets.float().to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # y = y.to(device).unsqueeze(1)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return dice_score


def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{os.path.basename(loader.dataset.data_list[idx])}"
        )
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        # torchvision.utils.save_image(y, f"{folder}{os.path.basename(loader.dataset.data_list[idx])}")

    model.train()
    return model


def train_2(module,traindataloader,valdataloader,device,num_epochs = 25):
    model=module.model
    optimizer=module.optimizer
    criterion=module.criterion
    scaler = torch.cuda.amp.GradScaler()

    source=0
    for epoch in range(num_epochs):
        print(f"-------------------现在开始第{epoch + 1}轮训练-------------------")
        train_fn(traindataloader, model, optimizer, criterion, scaler,device)

        # # save model
        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     "optimizer":optimizer.state_dict(),
        # }
        # save_checkpoint(checkpoint)

        # check accuracy
        dice_source=check_accuracy(valdataloader, model, device=device)

        # print some examples to a folder
        model=save_predictions_as_imgs(
            valdataloader, model, folder=f"output/{module.flag}/saved_images/", device=device
        )

        if dice_source>source:
            source=dice_source
            torch.save(model.state_dict(),f'output/{module.flag}/lzgd.pth')