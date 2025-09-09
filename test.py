import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset_test import DatasetFromFolder
from model import Generator, Discriminator
import utils
import argparse
import os
from stride_augmentation import *



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='crack500_test', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
params = parser.parse_args()
print(params)


data_dir = './dataset/crack500_test'
model_dir = './'
save_error_dir = r"./save"


if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(save_error_dir):
    os.mkdir(save_error_dir)

# Data pre-processing
test_transform = transforms.Compose([transforms.Resize(params.input_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Test data
test_data = DatasetFromFolder(data_dir, subfolder='test_folder_all', direction=params.direction, transform=test_transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)
G = Generator(3, params.ngf, 3)
D = Discriminator(6, params.ndf, 1)
D.cuda()
G.cuda()

G.load_state_dict(torch.load(model_dir + 'best_G_param.pkl'))

# Test
stride = False
for i, (input, target, label, input_name) in enumerate(test_data_loader):
    # input & target image data

    input_name = input_name[0]

    # print("input_name is is {}".format(input_name))

    input_np = (((input[0] - input[0].min()) * 255) / (input[0].max() - input[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)

    if stride:
        input_stride_all = []
        gen_image_all = []
        mask_all = []
        input_ori_all = []
        for j in range(12):
            # ratio_n = 0
            mask = random_stride_mask(input_np, ratio_n=j)
            mask_all.append(mask)
            input_ori = transforms.ToPILImage()(input_np)
            input_ori = test_transform(input_ori)
            input_ori = torch.unsqueeze(input_ori, 0)


            input_stride = input_np * mask
            input_stride = transforms.ToPILImage()(input_stride)



            input_stride = test_transform(input_stride)
            input_stride = torch.unsqueeze(input_stride, 0)

            input_stride_all.append(input_stride)

            input_ori_all.append(input_ori)

            x_ = Variable(input_stride.cuda())
            y_ = Variable(target.cuda())

            gen_image = G(x_)
            gen_image = gen_image.cpu().data
            gen_image_all.append(gen_image)
        concatenated_images = torch.cat(gen_image_all, dim=0)  # [12, 3, 256, 256]
        max_images = []
        for c in range(3):  # 遍历每个通道
            channel_images = concatenated_images[:, c, :, :]  # [12, 256, 256]
            max_channel_image = torch.max(channel_images, dim=0)[0]  # [256, 256]
            max_images.append(max_channel_image)
        max_image = torch.stack(max_images, dim=0)  # [3, 256, 256]
        max_image = max_image.unsqueeze(0)  # 增加一个维度，变为 [1, 3, 256, 256] # 形状为 [3, 256, 256]
        max_image = max_image.cpu().data

        utils.save_error_maps_gray(input_name, label, input_ori, target, max_image,
                                  i, save=True, save_dir=save_error_dir)


        # utils.save_error_maps_all(input_name, label, mask_all, input_ori_all, input_stride_all, target, gen_image_all, i, save=True, save_dir=save_error_dir)
        print('%d images are generated.' % (i + 1))

    else:
        input_ori = transforms.ToPILImage()(input_np)
        input_ori = test_transform(input_ori)
        
        input_ori = torch.unsqueeze(input_ori, 0)

        x_ = Variable(input_ori.cuda())
        y_ = Variable(target.cuda())

        gen_image = G(x_)
        gen_image = gen_image.cpu().data
        gen_image2 = gen_image[0].cpu().data.numpy()  # 转换为 NumPy 数组，并移至 CPU
        gen_image2 = gen_image2.transpose(1, 2, 0)  # 转换为 HWC 格式
        gen_image2 = (gen_image2 * 255).astype(np.uint8)  # 转换为 uint8 类型
        name = input_name.split("\\")[-1].split(".")[0]
        cv.imwrite("./result-crack/"+'{:s}'.format(name) + '.png', gen_image2)

        utils.save_error_maps_gray(input_name, label, input_ori, target, gen_image,
                                  i, save=True, save_dir=save_error_dir)
        print('%d images are generated.' % (i + 1))
