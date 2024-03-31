import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import optim


def original_SPGD(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, Nx=256, Ny=256):
    J_max = 0
    J = []
    gama_1 = 8
    gama_2 = 15
    epoch_num = 1500
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    for epoch in range(epoch_num):
        delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
        delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')

        add_a = a_0 + delta_a
        add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
        add_fhi = fhi_0 + delta_fhi
        add_complex = add_a * torch.exp(1j * add_fhi)

        sub_a = a_0 - delta_a
        sub_a = sub_a / torch.sqrt(torch.sum(torch.pow(sub_a, 2)))
        sub_fhi = fhi_0 - delta_fhi
        sub_complex = sub_a * torch.exp(1j * sub_fhi)

        add_E = torch.zeros(size=(Nx, Ny), device='cuda')
        sub_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_E = torch.squeeze(
                add_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(add_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_E

            sub_E = torch.squeeze(
                sub_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(sub_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + sub_E

        add_E = add_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_E), 2)))
        sub_E = sub_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(sub_E), 2)))

        add_J = torch.pow(
            torch.abs(
                torch.sum(add_E * torch.conj(E_target))
            ), 2)
        sub_J = torch.pow(
            torch.abs(
                torch.sum(sub_E * torch.conj(E_target))
            ), 2)

        delta_J = (add_J - sub_J) / 2
        # print(f'delta_J: {delta_J}')

        # J[index] = add_J.item()
        J.append(add_J.item())
        a_0 = a_0 + gama_1 * delta_a * delta_J
        a_0 = a_0 / torch.sqrt(
            torch.sum(
                torch.pow(a_0, 2)
            ))
        fhi_0 = fhi_0 + gama_2 * delta_fhi * delta_J

        if add_J.item() > J_max:
            J_max = add_J.item()
            complex_max = a_0 * torch.exp(1j * fhi_0)

        iteration = epoch
        if add_J.item() > 0.995:
            break

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)

    slm_cal_num = 2

    return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num


def lr_SPGD(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, final_value=3, epoch_num=1500, Nx=256, Ny=256,
            is_create_raw_data=False):
    J_max = 0
    J = []
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    for epoch in range(epoch_num):
        gama_1 = final_value + epoch_num / (1 + epoch)
        gama_2 = gama_1

        delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
        delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')

        add_a = a_0 + delta_a
        add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
        add_fhi = fhi_0 + delta_fhi
        add_complex = add_a * torch.exp(1j * add_fhi)

        sub_a = a_0 - delta_a
        sub_a = sub_a / torch.sqrt(torch.sum(torch.pow(sub_a, 2)))
        sub_fhi = fhi_0 - delta_fhi
        sub_complex = sub_a * torch.exp(1j * sub_fhi)

        add_E = torch.zeros(size=(Nx, Ny), device='cuda')
        sub_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_E = torch.squeeze(
                add_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(add_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_E

            sub_E = torch.squeeze(
                sub_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(sub_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + sub_E

        add_E = add_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_E), 2)))
        sub_E = sub_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(sub_E), 2)))

        add_J = torch.pow(
            torch.abs(
                torch.sum(add_E * torch.conj(E_target))
            ), 2)
        sub_J = torch.pow(
            torch.abs(
                torch.sum(sub_E * torch.conj(E_target))
            ), 2)

        delta_J = (add_J - sub_J) / 2

        # J[index] = add_J.item()
        J.append(add_J.item())
        a_0 = a_0 + gama_1 * delta_a * delta_J
        a_0 = a_0 / torch.sqrt(
            torch.sum(
                torch.pow(a_0, 2)
            ))
        fhi_0 = fhi_0 + gama_2 * delta_fhi * delta_J

        if add_J.item() > J_max:
            J_max = add_J.item()
            complex_max = a_0 * torch.exp(1j * fhi_0)

        iteration = epoch
        if add_J.item() > 0.995:
            break

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)

    slm_cal_num = 2

    if is_create_raw_data:
        return a_0, fhi_0, J_max, I_input.cpu(), I_output.cpu()
    else:
        return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num


def original_SGD(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, Nx=256, Ny=256):
    J_max = 0
    J = []
    gama_1 = 8
    gama_2 = 15
    epoch_num = 5000
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    for epoch in range(epoch_num):
        delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
        delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')

        # gradient a
        add_a = a_0 + delta_a
        add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
        add_complex = add_a * torch.exp(1j * fhi_0)
        original_complex = a_0 * torch.exp(1j * fhi_0)

        add_E = torch.zeros(size=(Nx, Ny), device='cuda')
        original_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_E = torch.squeeze(
                add_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(add_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_E

            original_E = torch.squeeze(
                original_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + original_E

        add_E = add_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_E), 2)))
        original_E = original_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(original_E), 2)))

        add_J = torch.pow(
            torch.abs(
                torch.sum(add_E * torch.conj(E_target))
            ), 2)
        original_J = torch.pow(
            torch.abs(
                torch.sum(original_E * torch.conj(E_target))
            ), 2)

        gradient_a = (add_J - original_J) / 2 * delta_a

        # gradient fhi
        add_fhi = fhi_0 + delta_fhi
        add_complex = a_0 * torch.exp(1j * add_fhi)
        original_complex = a_0 * torch.exp(1j * fhi_0)

        add_E = torch.zeros(size=(Nx, Ny), device='cuda')
        original_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_E = torch.squeeze(
                add_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_E

            original_E = torch.squeeze(
                original_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + original_E

        add_E = add_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_E), 2)))
        original_E = original_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(original_E), 2)))

        add_J = torch.pow(
            torch.abs(
                torch.sum(add_E * torch.conj(E_target))
            ), 2)
        original_J = torch.pow(
            torch.abs(
                torch.sum(original_E * torch.conj(E_target))
            ), 2)

        gradient_fhi = (add_J - original_J) / 2 * delta_fhi

        a_0 = a_0 + gama_1 * gradient_a
        a_0 = a_0 / torch.sqrt(
            torch.sum(
                torch.pow(a_0, 2)
            ))
        fhi_0 = fhi_0 + gama_2 * gradient_fhi
        upgrade_complex = a_0 * torch.exp(1j * fhi_0)
        upgrade_E = torch.zeros(size=(Nx, Ny), device='cuda')
        for i in range(mode_count):
            upgrade_E = torch.squeeze(
                upgrade_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + upgrade_E

        upgrade_E = upgrade_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(upgrade_E), 2)))
        upgrade_J = torch.pow(
            torch.abs(
                torch.sum(upgrade_E * torch.conj(E_target))
            ), 2)

        J.append(upgrade_J.item())
        if upgrade_J.item() > J_max:
            J_max = upgrade_J.item()
            complex_max = a_0 * torch.exp(1j * fhi_0)
        iteration = epoch
        if upgrade_J.item() > 0.995:
            break

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)

    slm_cal_num = 5

    return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num


def test_func(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, Nx=256, Ny=256):
    J_max = 0
    J = []
    gama_1 = 0.8
    gama_2 = 0.15
    epoch_num = 1500
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    for epoch in range(epoch_num):
        delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
        delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')

        add_a = a_0 + delta_a
        add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
        add_fhi = fhi_0 + delta_fhi
        add_complex = add_a * torch.exp(1j * add_fhi)

        add_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_E = torch.squeeze(
                add_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(add_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_E

        add_E = add_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_E), 2)))

        add_J = torch.pow(
            torch.abs(
                torch.sum(add_E * torch.conj(E_target))
            ), 2)

        delta_J = 1 - add_J

        # J[index] = add_J.item()
        J.append(add_J.item())
        a_0 = a_0 + gama_1 * delta_a * delta_J
        a_0 = a_0 / torch.sqrt(
            torch.sum(
                torch.pow(a_0, 2)
            ))
        fhi_0 = fhi_0 + gama_2 * delta_fhi * delta_J

        if add_J.item() > J_max:
            J_max = add_J.item()
            complex_max = a_0 * torch.exp(1j * fhi_0)

        iteration = epoch
        if add_J.item() > 0.995:
            break

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)

    slm_cal_num = 1

    return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num


def better_SGD(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, final_value=15, epoch_num=1200, Nx=256, Ny=256):
    J_max = 0
    J = []
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    for epoch in range(epoch_num):
        gama_1 = final_value + epoch_num / (1 + epoch)
        gama_2 = gama_1

        original_complex = a_0 * torch.exp(1j * fhi_0)
        original_E = torch.zeros(size=(Nx, Ny), device='cuda')
        for i in range(mode_count):
            original_E = torch.squeeze(
                original_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + original_E

        original_E = original_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(original_E), 2)))
        original_J = torch.pow(
            torch.abs(
                torch.sum(original_E * torch.conj(E_target))
            ), 2)

        J.append(original_J.item())
        if original_J.item() > J_max:
            J_max = original_J.item()
            complex_max = a_0 * torch.exp(1j * fhi_0)
        iteration = epoch
        if original_J.item() > 0.995:
            break

        delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
        delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')

        # gradient a
        add_a = a_0 + delta_a
        add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
        add_a_complex = add_a * torch.exp(1j * fhi_0)

        # gradient fhi
        add_fhi = fhi_0 + delta_fhi
        add_fhi_complex = a_0 * torch.exp(1j * add_fhi)

        add_a_E = torch.zeros(size=(Nx, Ny), device='cuda')
        add_fhi_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_a_E = torch.squeeze(
                add_a_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(add_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_a_E

            add_fhi_E = torch.squeeze(
                add_fhi_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_fhi_E

        add_a_E = add_a_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_a_E), 2)))

        add_a_J = torch.pow(
            torch.abs(
                torch.sum(add_a_E * torch.conj(E_target))
            ), 2)

        add_fhi_E = add_fhi_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_fhi_E), 2)))

        add_fhi_J = torch.pow(
            torch.abs(
                torch.sum(add_fhi_E * torch.conj(E_target))
            ), 2)

        gradient_a = (add_a_J - original_J) / 2 * delta_a
        gradient_fhi = (add_fhi_J - original_J) / 2 * delta_fhi

        a_0 = a_0 + gama_1 * gradient_a
        a_0 = a_0 / torch.sqrt(
            torch.sum(
                torch.pow(a_0, 2)
            ))
        fhi_0 = fhi_0 + gama_2 * gradient_fhi

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)

    slm_cal_num = 3

    return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num


def momentum_SGD(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, final_value=15, epoch_num=1200, momentum=0.1,
                 is_add=True, Nx=256, Ny=256):
    J_max = 0
    J = []
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    gradient_a = 0
    gradient_fhi = 0

    for epoch in range(epoch_num):
        gama_1 = final_value + epoch_num / (1 + epoch)
        gama_2 = gama_1

        original_complex = a_0 * torch.exp(1j * fhi_0)
        original_E = torch.zeros(size=(Nx, Ny), device='cuda')
        for i in range(mode_count):
            original_E = torch.squeeze(
                original_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + original_E

        original_E = original_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(original_E), 2)))
        original_J = torch.pow(
            torch.abs(
                torch.sum(original_E * torch.conj(E_target))
            ), 2)

        J.append(original_J.item())
        if original_J.item() > J_max:
            J_max = original_J.item()
            complex_max = a_0 * torch.exp(1j * fhi_0)
        iteration = epoch
        if original_J.item() > 0.995:
            break

        delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
        delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')

        # gradient a
        add_a = a_0 + delta_a
        add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
        add_a_complex = add_a * torch.exp(1j * fhi_0)

        # gradient fhi
        add_fhi = fhi_0 + delta_fhi
        add_fhi_complex = a_0 * torch.exp(1j * add_fhi)

        add_a_E = torch.zeros(size=(Nx, Ny), device='cuda')
        add_fhi_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_a_E = torch.squeeze(
                add_a_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(add_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_a_E

            add_fhi_E = torch.squeeze(
                add_fhi_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_fhi_E

        add_a_E = add_a_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_a_E), 2)))

        add_a_J = torch.pow(
            torch.abs(
                torch.sum(add_a_E * torch.conj(E_target))
            ), 2)

        add_fhi_E = add_fhi_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_fhi_E), 2)))

        add_fhi_J = torch.pow(
            torch.abs(
                torch.sum(add_fhi_E * torch.conj(E_target))
            ), 2)

        if is_add:
            gradient_a = (add_a_J - original_J) / 2 * delta_a + momentum * gradient_a
            gradient_fhi = (add_fhi_J - original_J) / 2 * delta_fhi + momentum * gradient_fhi
        else:
            gradient_a = (add_a_J - original_J) / 2 * delta_a - momentum * gradient_a
            gradient_fhi = (add_fhi_J - original_J) / 2 * delta_fhi - momentum * gradient_fhi

        a_0 = a_0 + gama_1 * gradient_a
        a_0 = a_0 / torch.sqrt(
            torch.sum(
                torch.pow(a_0, 2)
            ))
        fhi_0 = fhi_0 + gama_2 * gradient_fhi

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)

    slm_cal_num = 3

    return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num


def original_momentum_SGD(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, final_value=15, epoch_num=12000,
                          momentum_a=0.1, momentum_fhi=0.1,
                          is_add=True, Nx=256, Ny=256):
    J_max = 0
    J = []
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    gama_1 = 8
    gama_2 = 15

    gradient_a = 0
    gradient_fhi = 0

    for epoch in range(epoch_num):
        # gama_1 = final_value + epoch_num / (1 + epoch)
        # gama_2 = gama_1

        original_complex = a_0 * torch.exp(1j * fhi_0)
        original_E = torch.zeros(size=(Nx, Ny), device='cuda')
        for i in range(mode_count):
            original_E = torch.squeeze(
                original_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + original_E

        original_E = original_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(original_E), 2)))
        original_J = torch.pow(
            torch.abs(
                torch.sum(original_E * torch.conj(E_target))
            ), 2)

        J.append(original_J.item())
        if original_J.item() > J_max:
            J_max = original_J.item()
            complex_max = a_0 * torch.exp(1j * fhi_0)
        iteration = epoch
        if original_J.item() > 0.995:
            break

        delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
        delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')

        # gradient a
        add_a = a_0 + delta_a
        add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
        add_a_complex = add_a * torch.exp(1j * fhi_0)

        # gradient fhi
        add_fhi = fhi_0 + delta_fhi
        add_fhi_complex = a_0 * torch.exp(1j * add_fhi)

        add_a_E = torch.zeros(size=(Nx, Ny), device='cuda')
        add_fhi_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_a_E = torch.squeeze(
                add_a_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(add_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_a_E

            add_fhi_E = torch.squeeze(
                add_fhi_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_fhi_E

        add_a_E = add_a_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_a_E), 2)))

        add_a_J = torch.pow(
            torch.abs(
                torch.sum(add_a_E * torch.conj(E_target))
            ), 2)

        add_fhi_E = add_fhi_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_fhi_E), 2)))

        add_fhi_J = torch.pow(
            torch.abs(
                torch.sum(add_fhi_E * torch.conj(E_target))
            ), 2)

        if is_add:
            gradient_a = (1 - momentum_a) * (add_a_J - original_J) / 2 * delta_a + momentum_a * gradient_a
            gradient_fhi = (1 - momentum_fhi) * (add_fhi_J - original_J) / 2 * delta_fhi + momentum_fhi * gradient_fhi
        else:
            gradient_a = (1 - momentum_a) * (add_a_J - original_J) / 2 * delta_a - momentum_a * gradient_a
            gradient_fhi = (1 - momentum_fhi) * (add_fhi_J - original_J) / 2 * delta_fhi - momentum_fhi * gradient_fhi

        a_0 = a_0 + gama_1 * gradient_a
        a_0 = a_0 / torch.sqrt(
            torch.sum(
                torch.pow(a_0, 2)
            ))
        fhi_0 = fhi_0 + gama_2 * gradient_fhi

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)

    slm_cal_num = 3

    return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num


def test_func2(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, final_value=15, epoch_num=1200, momentum=0.5,
               Nx=256, Ny=256):
    J_max = 0
    J = []
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    gradient_a = 0
    gradient_fhi = 0

    for epoch in range(epoch_num):
        gama_1 = final_value + epoch_num / (1 + epoch)
        gama_2 = gama_1

        original_complex = a_0 * torch.exp(1j * fhi_0)
        original_E = torch.zeros(size=(Nx, Ny), device='cuda')
        for i in range(mode_count):
            original_E = torch.squeeze(
                original_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + original_E

        original_E = original_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(original_E), 2)))
        original_J = torch.pow(
            torch.abs(
                torch.sum(original_E * torch.conj(E_target))
            ), 2)

        J.append(original_J.item())
        if original_J.item() > J_max:
            J_max = original_J.item()
            complex_max = a_0 * torch.exp(1j * fhi_0)
        iteration = epoch
        if original_J.item() > 0.995:
            break

        delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
        delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')

        # gradient a
        add_a = a_0 + delta_a
        add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
        add_a_complex = add_a * torch.exp(1j * fhi_0)

        # gradient fhi
        add_fhi = fhi_0 + delta_fhi
        add_fhi_complex = a_0 * torch.exp(1j * add_fhi)

        add_a_E = torch.zeros(size=(Nx, Ny), device='cuda')
        add_fhi_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_a_E = torch.squeeze(
                add_a_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(add_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_a_E

            add_fhi_E = torch.squeeze(
                add_fhi_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_fhi_E

        add_a_E = add_a_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_a_E), 2)))

        add_a_J = torch.pow(
            torch.abs(
                torch.sum(add_a_E * torch.conj(E_target))
            ), 2)

        add_fhi_E = add_fhi_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_fhi_E), 2)))

        add_fhi_J = torch.pow(
            torch.abs(
                torch.sum(add_fhi_E * torch.conj(E_target))
            ), 2)

        gradient_a = (add_a_J - original_J) / 2 * delta_a - momentum * gradient_a
        gradient_fhi = (add_fhi_J - original_J) / 2 * delta_fhi - momentum * gradient_fhi

        a_0 = a_0 + gama_1 * gradient_a
        a_0 = a_0 / torch.sqrt(
            torch.sum(
                torch.pow(a_0, 2)
            ))
        fhi_0 = fhi_0 + gama_2 * gradient_fhi

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)

    slm_cal_num = 3

    return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num


def momentum_SGD_R(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, final_value=15, epoch_num=1200, momentum=0.1,
                   is_add=True, Nx=256, Ny=256):
    J_max = 0
    J = []
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    gradient_a = 0
    gradient_fhi = 0

    max_lr = 1500
    min_lr = 5

    for epoch in range(epoch_num):
        gama_1 = min_lr + 1 / 2 * (max_lr - min_lr) * (1 + np.cos(np.pi * (epoch % 200) / epoch_num))
        gama_2 = gama_1
        max_lr = 0.999 * max_lr
        min_lr = 0.999 * min_lr

        original_complex = a_0 * torch.exp(1j * fhi_0)
        original_E = torch.zeros(size=(Nx, Ny), device='cuda')
        for i in range(mode_count):
            original_E = torch.squeeze(
                original_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + original_E

        original_E = original_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(original_E), 2)))
        original_J = torch.pow(
            torch.abs(
                torch.sum(original_E * torch.conj(E_target))
            ), 2)

        J.append(original_J.item())
        if original_J.item() > J_max:
            J_max = original_J.item()
            complex_max = a_0 * torch.exp(1j * fhi_0)
        iteration = epoch
        if original_J.item() > 0.995:
            break

        delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
        delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')

        # gradient a
        add_a = a_0 + delta_a
        add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
        add_a_complex = add_a * torch.exp(1j * fhi_0)

        # gradient fhi
        add_fhi = fhi_0 + delta_fhi
        add_fhi_complex = a_0 * torch.exp(1j * add_fhi)

        add_a_E = torch.zeros(size=(Nx, Ny), device='cuda')
        add_fhi_E = torch.zeros(size=(Nx, Ny), device='cuda')

        for i in range(mode_count):
            add_a_E = torch.squeeze(
                add_a_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(add_a * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_a_E

            add_fhi_E = torch.squeeze(
                add_fhi_complex[i] * pro[i] / torch.sqrt(torch.sum(
                    torch.pow(a_0 * pro_r, 2))
                )
            ) * E_mode[i, :, :] + add_fhi_E

        add_a_E = add_a_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_a_E), 2)))

        add_a_J = torch.pow(
            torch.abs(
                torch.sum(add_a_E * torch.conj(E_target))
            ), 2)

        add_fhi_E = add_fhi_E / torch.sqrt(
            torch.sum(
                torch.pow(torch.abs(add_fhi_E), 2)))

        add_fhi_J = torch.pow(
            torch.abs(
                torch.sum(add_fhi_E * torch.conj(E_target))
            ), 2)

        if is_add:
            gradient_a = (add_a_J - original_J) / 2 * delta_a + momentum * gradient_a
            gradient_fhi = (add_fhi_J - original_J) / 2 * delta_fhi + momentum * gradient_fhi
        else:
            gradient_a = (add_a_J - original_J) / 2 * delta_a - momentum * gradient_a
            gradient_fhi = (add_fhi_J - original_J) / 2 * delta_fhi - momentum * gradient_fhi

        a_0 = a_0 + gama_1 * gradient_a
        a_0 = a_0 / torch.sqrt(
            torch.sum(
                torch.pow(a_0, 2)
            ))
        fhi_0 = fhi_0 + gama_2 * gradient_fhi

    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)

    slm_cal_num = 3

    return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num


def Adam_loss(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, Nx=256, Ny=256):
    complex_max = a_0 * torch.exp(1j * fhi_0)
    E = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E = torch.squeeze(
            complex_max[i] * pro[i] / torch.sqrt(torch.sum(
                torch.pow(a_0 * pro_r, 2))
            )
        ) * E_mode[i, :, :] + E
    E = E / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E), 2)))
    _J = torch.pow(
        torch.abs(
            torch.sum(E * torch.conj(E_target))
        ), 2)
    return 1 - _J


def original_Adam(a_0, fhi_0, pro, pro_r, E_mode, E_target, mode_count, Nx=256, Ny=256):
    Adam_a = a_0
    Adam_fhi = fhi_0
    Adam_a.requires_grad_(True)
    Adam_fhi.requires_grad_(True)
    J_max = 0
    J = []
    gama_1 = 8
    gama_2 = 15
    epoch_num = 2000
    iteration = 0
    complex_max = a_0 * torch.exp(1j * fhi_0)

    # for epoch in range(epoch_num):
    #     delta_a = torch.normal(0, 0.02, size=(mode_count, 1), device='cuda')
    #     delta_fhi = torch.normal(0, 0.3, size=(mode_count, 1), device='cuda')
    #
    #     add_a = a_0 + delta_a
    #     add_a = add_a / torch.sqrt(torch.sum(torch.pow(add_a, 2)))
    #     add_fhi = fhi_0 + delta_fhi
    #     add_complex = add_a * torch.exp(1j * add_fhi)
    #
    #     sub_a = a_0 - delta_a
    #     sub_a = sub_a / torch.sqrt(torch.sum(torch.pow(sub_a, 2)))
    #     sub_fhi = fhi_0 - delta_fhi
    #     sub_complex = sub_a * torch.exp(1j * sub_fhi)
    #
    #     add_E = torch.zeros(size=(Nx, Ny), device='cuda')
    #     sub_E = torch.zeros(size=(Nx, Ny), device='cuda')
    #
    #     for i in range(mode_count):
    #         add_E = torch.squeeze(
    #             add_complex[i] * pro[i] / torch.sqrt(torch.sum(
    #                 torch.pow(add_a * pro_r, 2))
    #             )
    #         ) * E_mode[i, :, :] + add_E
    #
    #         sub_E = torch.squeeze(
    #             sub_complex[i] * pro[i] / torch.sqrt(torch.sum(
    #                 torch.pow(sub_a * pro_r, 2))
    #             )
    #         ) * E_mode[i, :, :] + sub_E
    #
    #     add_E = add_E / torch.sqrt(
    #         torch.sum(
    #             torch.pow(torch.abs(add_E), 2)))
    #     sub_E = sub_E / torch.sqrt(
    #         torch.sum(
    #             torch.pow(torch.abs(sub_E), 2)))
    #
    #     add_J = torch.pow(
    #         torch.abs(
    #             torch.sum(add_E * torch.conj(E_target))
    #         ), 2)
    #     sub_J = torch.pow(
    #         torch.abs(
    #             torch.sum(sub_E * torch.conj(E_target))
    #         ), 2)
    #
    #     delta_J = (add_J - sub_J) / 2
    #     # print(f'delta_J: {delta_J}')
    #
    #     # J[index] = add_J.item()
    #     J.append(add_J.item())
    #     a_0 = a_0 + gama_1 * delta_a * delta_J
    #     a_0 = a_0 / torch.sqrt(
    #         torch.sum(
    #             torch.pow(a_0, 2)
    #         ))
    #     fhi_0 = fhi_0 + gama_2 * delta_fhi * delta_J
    #
    #     if add_J.item() > J_max:
    #         J_max = add_J.item()
    #         complex_max = a_0 * torch.exp(1j * fhi_0)
    #
    #     iteration = epoch
    #     if add_J.item() > 0.995:
    #         break
    #
    # E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    # E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    # for i in range(mode_count):
    #     E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
    #     E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input
    #
    # E_output = E_output / torch.sqrt(
    #     torch.sum(
    #         torch.pow(torch.abs(E_output), 2)))
    # E_input = E_input / torch.sqrt(
    #     torch.sum(
    #         torch.pow(torch.abs(E_input), 2)))
    #
    # I_output = torch.pow(torch.abs(E_output), 2)
    # I_input = torch.pow(torch.abs(E_input), 2)
    #
    # slm_cal_num = 2
    #
    # return I_input.cpu(), I_output.cpu(), J, J_max, iteration, slm_cal_num
    print(Adam_loss(Adam_a, Adam_fhi, pro, pro_r, E_mode, E_target, mode_count))

    optimizer = optim.Adam([Adam_a, Adam_fhi], lr=0.01)
    J_value = 0
    while epoch_num > 0:
        loss_value = Adam_loss(Adam_a, Adam_fhi, pro, pro_r, E_mode, E_target, mode_count)
        print(f'J: {loss_value.item()}')
        J.append(-loss_value.item() + 1)
        loss_value.backward()
        optimizer.step()
        epoch_num -= 1

    complex_max = a_0 * torch.exp(1j * fhi_0)
    E_output = torch.zeros(size=(Nx, Ny), device='cuda')
    E_input = torch.zeros(size=(Nx, Ny), device='cuda')
    for i in range(mode_count):
        E_output = torch.squeeze(complex_max[i] * pro[i] * E_mode[i, :, :]) + E_output
        E_input = torch.squeeze(complex_max[i] * E_mode[i, :, :]) + E_input

    E_output = E_output / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_output), 2)))
    E_input = E_input / torch.sqrt(
        torch.sum(
            torch.pow(torch.abs(E_input), 2)))

    I_output = torch.pow(torch.abs(E_output), 2)
    I_input = torch.pow(torch.abs(E_input), 2)
    slm_cal_num = 2
    title = 'Adam'
    plt.figure()
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.imshow(I_input.cpu().detach().numpy(), cmap='jet')
    plt.title('Pre-shaped input')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(I_output.cpu().detach().numpy(), cmap='jet')
    plt.title('Shaped output')
    plt.axis('off')
    plt.figure()
    plt.plot(J)
