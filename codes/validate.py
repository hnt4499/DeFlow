import argparse
import os

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

from tqdm import tqdm

import torch
import numpy as np
import imageio

import MeasureLib


def rgb(t, denormalize_f):
    t = (t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0])
    return (np.clip(denormalize_f(t), 0, 1) * 255).round().astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-opt', required=True, help='path to config .yml file')
    parser.add_argument('-model_path', required=True, help='path to model checkpoint (.pkl)')
    parser.add_argument('-out_dir', default=None, help='save directory of translated images')
    parser.add_argument('-crop_size', type=int, default=256, help='')
    parser.add_argument('-n_max', type=int, default=None, help='number of images to translate, if not specified all validation images are translated')
    add_gt_noise = True # apply quantization noise
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = f'../results/{args.opt[:-4]}-{args.model_path.split("/")[-1][:-4]}{"-"+str(args.crop_size) if args.crop_size else ""}/'

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_dir+'/0_to_1', exist_ok=True)
    os.makedirs(args.out_dir+'/1_to_0', exist_ok=True)
    os.makedirs(args.out_dir+'/0_gen', exist_ok=True)
    os.makedirs(args.out_dir+'/1_gen', exist_ok=True)

    model = create_model(util.getPredictorOdict(os.path.join("confs/", args.opt), args.model_path))
    opt = option.parse("confs/" + args.opt, is_train=True)
    device = model.netG.module.mean_shift.device

    dataset_opt = opt['datasets']['val']
    dataset_opt['center_crop_hr_size'] = args.crop_size
    dataset_opt['n_max'] = args.n_max
    dataset_opt['preload'] = False
    dataset_opt['phase'] = "test"

    val_set = create_dataset(dataset_opt)
    val_loader = create_dataloader(val_set, dataset_opt, opt, None)

    t = "train" if "train" in opt['datasets'] else "train_0"
    if 'normalize' in opt['datasets'][t]:
        mean_noisy_hr = np.asarray(opt['datasets'][t]['normalize']['mean_noisy_hr'])/255
        std_noisy_hr = np.asarray(opt['datasets'][t]['normalize']['std_noisy_hr'])/255
        mean_clean_hr = np.asarray(opt['datasets'][t]['normalize']['mean_clean_hr'])/255
        std_clean_hr = np.asarray(opt['datasets'][t]['normalize']['std_clean_hr'])/255
        denormalize_domY = lambda x: (((x - mean_clean_hr)/std_clean_hr)*std_noisy_hr) + mean_noisy_hr
    else:
        denormalize_domY = lambda x: x

    noisy_gt, noisy_syn = [], []  # store synthetically degraded and given degraded images for metric evaluation

    for idx_val, val_data in tqdm(enumerate(val_loader), total=len(val_loader)):
        lq, gt, labels = val_data['LQ'], val_data['GT'], val_data['y_label']

        lq, gt, labels = lq.to(device), gt.to(device), labels.to(device)

        fn = val_data['GT_path'][0].split('/')[-1][:-4]
        save_path = os.path.join(args.out_dir, '{}_to_{}/{}_original.png'.format(labels.item(), 1-labels.item(),fn))
        denormalize_f = denormalize_domY if labels[0] == 1 else lambda x: x
        gt_denormalized = rgb(gt, denormalize_f)
        imageio.imwrite(save_path, gt_denormalized)

        if labels[0] == 1:
            noisy_gt.append(gt_denormalized)

        # endcode
        lr_enc = model.netG.module.rrdbPreprocessing(lq) # precomute lr encoding s.t. conditional features are fixed!
        zs, nll = model.get_encode_z_and_nll(lq=lq, gt=gt, y_label=labels, lr_enc=lr_enc, add_gt_noise=add_gt_noise, epses=[])

        # translate latents zs and decode
        translated = model.get_translate_with_zs(zs=zs, lq=gt, source_labels=labels, lr_enc=lr_enc, heat=1.0)
        save_path = os.path.join(args.out_dir,
                                    '{}_to_{}/{}_translated.png'.format(labels.item(), 1-labels.item(),fn))
        denormalize_f = denormalize_domY if labels[0] == 0 else lambda x: x
        translated_denormalized = rgb(translated, denormalize_f)
        imageio.imwrite(save_path, translated_denormalized)

        if labels[0] == 0:
            noisy_syn.append(translated_denormalized)

        # domain wise conditional generation
        # p_x(.|h(x))
        sr0 = model.get_sr(lq=lq, lr_enc=lr_enc, y_label=torch.tensor([0]), heat=1.0)
        save_path = os.path.join(args.out_dir, '0_gen/{}_{}_generated.png'.format(fn, labels.item()))
        imageio.imwrite(save_path, rgb(sr0, lambda x: x))

        # p_y(.|h(y))
        sr1 = model.get_sr(lq=lq, lr_enc=lr_enc, y_label=torch.tensor([1]), heat=1.0)
        save_path = os.path.join(args.out_dir, '1_gen/{}_{}_generated.png'.format(fn, labels.item()))
        imageio.imwrite(save_path, rgb(sr1, denormalize_domY))

    # Starting evaluation
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    assert len(noisy_gt) == len(noisy_syn)

    for noisy_gt_i, noisy_syn_i in zip(noisy_gt, noisy_syn):
        psnr, ssim, lpips = MeasureLib.measure(noisy_gt_i, noisy_syn_i)

        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lpips

    avg_psnr = avg_psnr / len(noisy_gt)
    avg_ssim = avg_ssim / len(noisy_gt)
    avg_lpips = avg_lpips / len(noisy_gt)
    result_str = f"Evaluation results: psnr: {avg_psnr:.4e}, ssim: {avg_ssim:.4e}, lpips: {avg_lpips:.4e}"
    print(result_str)

    with open(os.path.join(args.out_dir, "evaluation_results.txt"), "w") as fout:
        fout.write(result_str + "\n")