import argparse
import glob
import os
import re
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from skimage.measure import label
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=os.path.abspath(os.path.join(
                        SCRIPT_DIR, '..', 'data', 'PROMISE12_h5')),
                    help='dataset root path')
parser.add_argument('--exp', type=str,
                    default='MT_PROMISE12_UniMatch_MRIInterp',
                    help='experiment name, must match training --exp. If the checkpoint is not found, the script will also auto-search.')
parser.add_argument('--model', type=str,
                    default='unet',
                    help='model name')
parser.add_argument('--num_classes', type=int,
                    default=2,
                    help='number of classes, PROMISE12 is binary')
parser.add_argument('--labelnum', type=int,
                    default=7,
                    help='number of labeled PROMISE12 cases, must match training')
parser.add_argument('--stage_name', type=str,
                    default='self_train',
                    choices=['pre_train', 'self_train'],
                    help='load model from pre_train or self_train')
parser.add_argument('--patch_size', type=int, nargs=2,
                    default=[256, 256],
                    help='network input patch size')
parser.add_argument('--gpu', type=str,
                    default='0',
                    help='GPU id')
parser.add_argument('--save_result', type=str,
                    default='True',
                    choices=['True', 'False', 'true', 'false', '1', '0', 'yes', 'no'],
                    help='whether to save prediction nii.gz files')
parser.add_argument('--nms', type=int,
                    default=0,
                    help='whether to keep the largest 3D connected foreground component')
parser.add_argument('--model_root', type=str,
                    default='../model',
                    help='model root. Relative paths are resolved relative to this script file, not the current working directory.')
parser.add_argument('--snapshot_path', type=str,
                    default=None,
                    help='optional snapshot path. If set, it overrides model_root/exp/labelnum/stage/model')
parser.add_argument('--checkpoint_path', type=str,
                    default=None,
                    help='optional exact checkpoint path. If set, it has highest priority')
parser.add_argument('--auto_find_checkpoint', type=str,
                    default='True',
                    choices=['True', 'False', 'true', 'false', '1', '0', 'yes', 'no'],
                    help='if the expected checkpoint is missing, automatically find best/latest checkpoint under model_root')
parser.add_argument('--test_save_path', type=str,
                    default=None,
                    help='optional prediction save path. Default: snapshot_path/test_predictions')
FLAGS = parser.parse_args()


def str2bool(v):
    return str(v).lower() in ['true', '1', 'yes', 'y']


def resolve_path(path, base_dir=SCRIPT_DIR):
    if path is None or len(str(path)) == 0:
        return path
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def calculate_metric_percase(pred, gt):
    """Return Dice, Jaccard, HD95, ASD for one binary foreground class."""
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0, 0, 0, 0
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0, 0, 0, 0
    else:
        return 1, 1, 0, 0


def get_largest_cc(segmentation):
    labels = label(segmentation)
    if labels.max() == 0:
        return segmentation
    largest_cc = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)
    return largest_cc.astype(segmentation.dtype)


def apply_nms_postprocessing(prediction):
    processed = np.zeros_like(prediction, dtype=prediction.dtype)
    fg = prediction == 1
    if fg.sum() > 0:
        fg = get_largest_cc(fg)
        processed[fg > 0] = 1
    return processed


def build_model(model_name, in_chns, class_num):
    try:
        from networks.net_factory import net_factory
        net = net_factory(net_type=model_name, in_chns=in_chns, class_num=class_num)
        if net is not None:
            return net
    except Exception as e:
        print("[Warning] net_factory failed for model '{}': {}".format(model_name, e))

    if model_name == 'unet':
        from networks.unet import UNet_2d
        return UNet_2d(in_chns=in_chns, class_num=class_num)

    if model_name == 'unet_xingtai':
        from networks.unet_xingtai import UNet_XingTai
        return UNet_XingTai(in_chns=in_chns, class_num=class_num)

    raise ValueError("Unknown model name: {}".format(model_name))


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v
    return new_state


def extract_state_dict(checkpoint, model_path):
    """Support all checkpoint formats used in your training scripts.

    Supported formats:
    1) raw model.state_dict()
    2) {'net': model.state_dict(), 'opt': optimizer.state_dict()}
    3) {'state_dict': model.state_dict(), ...}
    4) {'model': model.state_dict(), ...}
    """
    if isinstance(checkpoint, dict):
        for key in ['net', 'state_dict', 'model', 'model_state_dict']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                print("Load checkpoint['{}'] from {}".format(key, model_path))
                return checkpoint[key]

        # Raw state_dict is also a dict whose values are tensors/parameters.
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            print('Load raw state_dict from {}'.format(model_path))
            return checkpoint

    raise RuntimeError('Unsupported checkpoint format: {}'.format(model_path))


def load_model_weight(net, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model file not found: {}'.format(model_path))

    try:
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    state_dict = extract_state_dict(checkpoint, model_path)
    state_dict = strip_module_prefix(state_dict)

    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print('[Warning] strict=True loading failed. Trying strict=False.')
        print(str(e))
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print('[Warning] Missing keys:', missing)
        if len(unexpected) > 0:
            print('[Warning] Unexpected keys:', unexpected)
    return net


def parse_iter_num(path):
    name = os.path.basename(path)
    m = re.search(r'iter_(\d+)', name)
    return int(m.group(1)) if m else -1


def parse_dice(path):
    name = os.path.basename(path)
    m = re.search(r'dice_([0-9.]+)', name)
    if m:
        try:
            return float(m.group(1).rstrip('.'))
        except Exception:
            return -1.0
    return -1.0


def get_snapshot_path(flags):
    if flags.snapshot_path is not None and len(flags.snapshot_path) > 0:
        return resolve_path(flags.snapshot_path)

    model_root = resolve_path(flags.model_root)
    return os.path.join(
        model_root,
        '{}_{}_labeled'.format(flags.exp, flags.labelnum),
        flags.stage_name,
        flags.model
    )


def rank_checkpoint_candidates(candidates):
    """Prefer best checkpoint, then highest dice checkpoint, then latest iter, then newest mtime."""
    unique = []
    seen = set()
    for p in candidates:
        p = os.path.abspath(p)
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            unique.append(p)

    def key_fn(p):
        base = os.path.basename(p)
        is_best = 1 if base.endswith('_best_model.pth') or 'best' in base else 0
        dice = parse_dice(p)
        it = parse_iter_num(p)
        mtime = os.path.getmtime(p)
        return (is_best, dice, it, mtime)

    return sorted(unique, key=key_fn, reverse=True)


def find_checkpoint(flags, snapshot_path):
    tried = []

    if flags.checkpoint_path is not None and len(flags.checkpoint_path) > 0:
        ckpt = resolve_path(flags.checkpoint_path)
        tried.append(ckpt)
        if os.path.exists(ckpt):
            return ckpt, tried
        raise FileNotFoundError('Explicit --checkpoint_path does not exist: {}'.format(ckpt))

    expected = os.path.join(snapshot_path, '{}_best_model.pth'.format(flags.model))
    tried.append(expected)
    if os.path.exists(expected):
        return expected, tried

    if not str2bool(flags.auto_find_checkpoint):
        raise FileNotFoundError('Model file not found: {}'.format(expected))

    candidates = []
    # 1) Search current expected snapshot first.
    candidates.extend(glob.glob(os.path.join(snapshot_path, '*.pth')))

    # 2) Search the same experiment under model_root, in case stage/model name is mismatched.
    model_root = resolve_path(flags.model_root)
    exp_dir = os.path.join(model_root, '{}_{}_labeled'.format(flags.exp, flags.labelnum))
    candidates.extend(glob.glob(os.path.join(exp_dir, '**', '*.pth'), recursive=True))

    # 3) Search all folders that contain the exp string. This helps when the default exp differs from training exp.
    pattern = os.path.join(model_root, '*{}*{}_labeled'.format(flags.exp, flags.labelnum), '**', '*.pth')
    candidates.extend(glob.glob(pattern, recursive=True))

    # 4) Last fallback: search PROMISE12/Prostate/LARI/MT folders with the same label number.
    broad_patterns = [
        os.path.join(model_root, '*PROMISE12*{}_labeled'.format(flags.labelnum), '**', '*.pth'),
        os.path.join(model_root, '*LARI*{}_labeled'.format(flags.labelnum), '**', '*.pth'),
        os.path.join(model_root, '*MT*Prostate*{}_labeled'.format(flags.labelnum), '**', '*.pth'),
        os.path.join(model_root, '*Prostate*{}_labeled'.format(flags.labelnum), '**', '*.pth'),
    ]
    for pat in broad_patterns:
        candidates.extend(glob.glob(pat, recursive=True))

    ranked = rank_checkpoint_candidates(candidates)
    tried.extend(ranked[:20])
    if len(ranked) > 0:
        print('[Auto-find] Expected checkpoint is missing: {}'.format(expected))
        print('[Auto-find] Use checkpoint: {}'.format(ranked[0]))
        print('[Auto-find] Top candidates:')
        for p in ranked[:8]:
            print('  - {}'.format(p))
        return ranked[0], tried

    message = ['Model file not found. Tried:']
    for p in tried:
        message.append('  - {}'.format(p))
    message.append('')
    message.append('Please either:')
    message.append('  1) set --exp to the exact training exp, e.g. --exp MT_PROMISE12_baseline')
    message.append('  2) set --checkpoint_path to the exact .pth file')
    message.append('  3) set --model_root to the correct ../model directory')
    raise FileNotFoundError('\n'.join(message))


def test_single_volume(case, net, test_save_path, flags, device):
    h5_path = os.path.join(flags.root_path, 'data', '{}.h5'.format(case))
    if not os.path.exists(h5_path):
        raise FileNotFoundError('Case file not found: {}'.format(h5_path))

    with h5py.File(h5_path, 'r') as h5f:
        image = h5f['image'][:]
        label_gt = h5f['label'][:]

    prediction = np.zeros_like(label_gt)
    net.eval()

    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]

        slice_resized = zoom(
            slice_img,
            (flags.patch_size[0] / x, flags.patch_size[1] / y),
            order=1,
            prefilter=False
        )

        input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out_main = net(input_tensor)
            if isinstance(out_main, (tuple, list)):
                out_main = out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

        pred = zoom(
            out,
            (x / flags.patch_size[0], y / flags.patch_size[1]),
            order=0
        )
        prediction[ind] = pred.astype(label_gt.dtype)

    if flags.nms == 1:
        prediction = apply_nms_postprocessing(prediction)

    first_metric = calculate_metric_percase(prediction == 1, label_gt == 1)

    if str2bool(flags.save_result):
        os.makedirs(test_save_path, exist_ok=True)

        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.SetSpacing((1, 1, 10))

        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.SetSpacing((1, 1, 10))

        lab_itk = sitk.GetImageFromArray(label_gt.astype(np.float32))
        lab_itk.SetSpacing((1, 1, 10))

        sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + '_pred.nii.gz'))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, case + '_img.nii.gz'))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + '_gt.nii.gz'))

    return first_metric


def inference(flags):
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    list_path = os.path.join(flags.root_path, 'test.list')
    if not os.path.exists(list_path):
        raise FileNotFoundError('test.list not found: {}'.format(list_path))

    with open(list_path, 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split('.')[0] for item in image_list])
    if len(image_list) == 0:
        raise RuntimeError('No test cases found in {}'.format(list_path))

    snapshot_path = get_snapshot_path(flags)
    checkpoint_path, tried_paths = find_checkpoint(flags, snapshot_path)

    if flags.test_save_path is not None and len(flags.test_save_path) > 0:
        test_save_path = resolve_path(flags.test_save_path)
    else:
        test_save_path = os.path.join(snapshot_path, 'test_predictions')

    performance_path = os.path.join(snapshot_path, 'performance.txt')

    if str2bool(flags.save_result):
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path, exist_ok=True)

    os.makedirs(snapshot_path, exist_ok=True)

    net = build_model(flags.model, in_chns=1, class_num=flags.num_classes)
    net = net.to(device)
    net = load_model_weight(net, checkpoint_path, device)
    net = net.to(device)
    net.eval()

    print('Init weight from {}'.format(checkpoint_path))
    print('Snapshot path: {}'.format(snapshot_path))
    print('Model root: {}'.format(resolve_path(flags.model_root)))
    print('NMS post-processing: {}'.format('ON' if flags.nms == 1 else 'OFF'))
    print('Save results: {}'.format(str2bool(flags.save_result)))

    total_metric = np.zeros(4, dtype=np.float64)
    case_metric_dict = {}

    for case in tqdm(image_list):
        case_metric = test_single_volume(case, net, test_save_path, flags, device)
        case_metric = np.asarray(case_metric, dtype=np.float64)
        total_metric += case_metric
        case_metric_dict[case] = case_metric

        print('{} -> Dice: {:.4f}, Jaccard: {:.4f}, HD95: {:.4f}, ASD: {:.4f}'.format(
            case, case_metric[0], case_metric[1], case_metric[2], case_metric[3]
        ))

    avg_metric = total_metric / len(image_list)

    with open(performance_path, 'w') as f:
        f.write('Model path: {}\n'.format(checkpoint_path))
        f.write('Snapshot path: {}\n'.format(snapshot_path))
        f.write('Model root: {}\n'.format(resolve_path(flags.model_root)))
        f.write('Model: {}\n'.format(flags.model))
        f.write('Experiment: {}\n'.format(flags.exp))
        f.write('Stage: {}\n'.format(flags.stage_name))
        f.write('NMS: {}\n'.format(flags.nms))
        f.write('Test cases: {}\n\n'.format(len(image_list)))

        for case in image_list:
            m = case_metric_dict[case]
            f.write('{} -> Dice: {:.6f}, Jaccard: {:.6f}, HD95: {:.6f}, ASD: {:.6f}\n'.format(
                case, m[0], m[1], m[2], m[3]
            ))

        f.write('\nAverage metric:\n')
        f.write('Dice: {:.6f}\n'.format(avg_metric[0]))
        f.write('Jaccard: {:.6f}\n'.format(avg_metric[1]))
        f.write('HD95: {:.6f}\n'.format(avg_metric[2]))
        f.write('ASD: {:.6f}\n'.format(avg_metric[3]))

    return avg_metric, test_save_path, performance_path


if __name__ == '__main__':
    metric_avg, save_path, perf_path = inference(FLAGS)
    print('\n================ Final Average Metrics ================')
    print('Dice:    {:.6f}'.format(metric_avg[0]))
    print('Jaccard: {:.6f}'.format(metric_avg[1]))
    print('HD95:    {:.6f}'.format(metric_avg[2]))
    print('ASD:     {:.6f}'.format(metric_avg[3]))
    print('Prediction save path: {}'.format(save_path))
    print('Performance file: {}'.format(perf_path))
