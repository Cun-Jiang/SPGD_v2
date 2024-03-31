# # 这是一个示例 Python 脚本。
#
# # 按 Shift+F10 执行或将其替换为您的代码。
# # 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
#
#
# def print_hi(name):
#     # 在下面的代码行中使用断点来调试脚本。
#     print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
#
#
# # 按装订区域中的绿色按钮以运行脚本。
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

import torch
import time
import matplotlib.pyplot as plt
from optimizeFunc import lr_SPGD
from itertools import combinations


def random_data_img_generate(pro, E_mode, mode_count, epoch=0, Nx=256, Ny=256):
    _a = torch.rand(size=(mode_count, 1))
    _a = _a / torch.sqrt(torch.sum(torch.pow(_a, 2)))
    _fhi = torch.rand(size=(mode_count, 1)) * 2 * torch.pi
    _complex = _a * torch.exp(1j * _fhi)

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')

    for i in range(mode_count):
        E_output = torch.squeeze(_complex[i] * pro[i] * E_mode[i, :, :]) + E_output

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))

    I_output = torch.pow(torch.abs(E_output), 2).cpu()

    torch.save({
        'a': _a,
        'fhi': _fhi,
        'E_output': E_output,
    }, f'./data_1/doc/epoch_{epoch}_data.pth')

    plt.figure()
    plt.axis('off')
    plt.imshow(I_output, cmap='jet')
    plt.savefig(f'./data_1/img/epoch_{epoch}_output_img.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # return I_input.cpu(), I_output.cpu()


torch.set_default_device('cuda')
init_data = torch.load('./data_1/init_data.pth', map_location=torch.device('cuda:0'))

pro_r = init_data['pro_r']
pro = init_data['pro']
E_mode = init_data['E_mode']
mode_count = init_data['modeCount']
epoch_num = 36000

for epoch in range(epoch_num):
    begin = time.time()

    random_data_img_generate(pro, E_mode, mode_count, epoch)

    end = time.time()
    if epoch % 500 == 0:
        print(f'epoch: {epoch}, run time: {end - begin} s')
