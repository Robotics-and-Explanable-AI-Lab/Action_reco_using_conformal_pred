#!/usr/bin/env python3
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2
import csv

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import (GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor,
                            GroupNormalize, GroupFullResSample, GroupOverSample)
from ops import dataset_config

# ------------------------------
# Utility Functions and Classes
# ------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                shift_div = int(s.replace('shift', ''))
                shift_place = strings[i + 1] if i + 1 < len(strings) else ''
                return True, shift_div, shift_place
    return False, None, None

def combine_p_values_fisher(p_values):
    p_values = np.maximum(np.array(p_values), 1e-8)
    fisher_stat = -2.0 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    combined_p = chi2.sf(fisher_stat, df)
    return combined_p

# ------------------------------
# Conformal Predictor Class
# ------------------------------
class SingleLabelConformalPredictor:
    def __init__(self, model, significance=0.05, device='cuda'):
        """
        Initialize the conformal predictor using ICAD.
        :param model: A pretrained model that implements predict_proba().
        :param significance: Significance level (e.g., 0.05 for 95% confidence).
        :param device: Device for computation.
        """
        self.model = model
        self.significance = significance
        self.device = device
        self.calib_scores = None  # nonconformity scores from calibration
        self.threshold = None     # threshold computed from calibration

    def calibrate(self, calib_loader):
        scores = []
        self.model.eval()
        with torch.no_grad():
            for segments, labels, *rest in calib_loader:
                segments = segments.to(self.device)
                labels = labels.to(self.device)
                with autocast():
                    probs = self.model.predict_proba(segments)  # (B, num_classes)
                probs = torch.tensor(probs, device=self.device)
                for i in range(probs.size(0)):
                    true_label = int(labels[i].item())
                    true_prob = probs[i, true_label].item()
                    score = 1.0 - true_prob
                    scores.append(score)
        scores = np.array(scores)
        self.calib_scores = np.sort(scores)
        self.threshold = np.quantile(self.calib_scores, 1.0 - self.significance)
        print(f"[CP] Calibration complete on {len(scores)} samples, threshold = {self.threshold:.4f}")

    def evaluate(self, eval_loader):
        """
        Evaluate the conformal predictor on a test set.
        Additionally, compute Top-5 accuracy.
        :param eval_loader: DataLoader yielding (segments, labels, ...).
        :return: A tuple (results, top5_acc) where results is a list of dictionaries 
                 and top5_acc is the overall Top-5 accuracy.
        """
        results = []
        top5_correct = 0
        top5_total = 0
        self.model.eval()
        with torch.no_grad():
            for segments, labels, *rest in eval_loader:
                # Attempt to extract file name/info from extra returned values
                if rest and len(rest) > 0:
                    file_info = rest[0]
                else:
                    file_info = "N/A"
                segments = segments.to(self.device)
                with autocast():
                    probs = self.model.predict_proba(segments)
                probs = torch.tensor(probs, device=self.device)
                for i in range(probs.size(0)):
                    # Compute Top-5 predictions
                    topk_vals, topk_indices = torch.topk(probs[i], 5, dim=0)
                    pred_label = int(topk_indices[0].item())
                    # Update Top-5 accuracy
                    true_label = int(labels[i].item())
                    if true_label in topk_indices.cpu().numpy():
                        top5_correct += 1
                    top5_total += 1
                    conf = probs[i, pred_label].item()
                    nonconformity = 1.0 - conf
                    idx = np.searchsorted(self.calib_scores, nonconformity, side='left')
                    num_ge = len(self.calib_scores) - idx
                    p_value = (num_ge + 1) / (len(self.calib_scores) + 1)
                    # If file_info is list-like, get the corresponding file name; otherwise use as is.
                    if isinstance(file_info, (list, tuple)) and len(file_info) > i:
                        fname = file_info[i]
                    else:
                        fname = file_info
                    results.append({
                        "file_name": fname,
                        "predicted_label": pred_label,
                        "true_label": true_label,
                        "confidence_score": conf,
                        "nonconformity": nonconformity,
                        "p_value": p_value,
                        f"accepted_at_{int((1-self.significance)*100)}%": nonconformity <= self.threshold
                    })
        top5_acc = top5_correct / top5_total if top5_total > 0 else 0
        return results, top5_acc

# ------------------------------
# Argument Parsing
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Assembly101 TSN CP Evaluation")
    parser.add_argument("--root_path", type=str, required=True, help="Path to dataset root folder.")
    parser.add_argument("--calib_list", type=str, required=True, help="Path to the calibration list file (Validation).")
    parser.add_argument("--eval_list", type=str, required=True, help="Path to the evaluation list file.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the pretrained weights (.pth.tar).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading.")
    parser.add_argument("--num_segments", type=int, default=8, help="Number of segments to sample.")
    parser.add_argument("--input_size", type=int, default=224, help="Input size (height/width)")
    parser.add_argument("--significance", type=float, default=0.05, help="Significance level for CP")
    parser.add_argument("--test_segments", type=str, default="8", help="Comma-separated segment counts for testing")
    parser.add_argument("--dense_sample", default=False, action="store_true", help="Use dense sampling")
    parser.add_argument("--twice_sample", default=False, action="store_true", help="Use twice sampling for ensemble")
    parser.add_argument("--full_res", default=False, action="store_true", help="Use full resolution (256x256)")
    parser.add_argument("--test_crops", type=int, default=1, help="Number of test crops (1, 5, or 10)")
    parser.add_argument("--coeff", type=str, default=None, help="Comma-separated coefficients for ensemble")
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N", help="Number of data loading workers")
    parser.add_argument("--text_file", type=str, default=None, help="Optional text file for splits")
    parser.add_argument("--softmax", default=False, action="store_true", help="Apply softmax to model output")
    parser.add_argument("--max_num", type=int, default=-1, help="Max number of samples to evaluate")
    parser.add_argument("--crop_fusion_type", type=str, default="avg", help="Crop fusion type")
    parser.add_argument("--gpus", nargs="+", type=int, default=None, help="List of GPU IDs")
    parser.add_argument("--img_feature_dim", type=int, default=256, help="Image feature dimension")
    parser.add_argument("--num_set_segments", type=int, default=1, help="Multiple sets of n-frames from a video")
    parser.add_argument("--pretrain", type=str, default="imagenet", help="Pretrain option")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., Assembly101)")
    return parser.parse_args()

# ------------------------------
# Main Function
# ------------------------------
def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup ensemble if multiple weights are provided
    weights_list = args.weights.split(',')
    test_segments_list = [int(s) for s in args.test_segments.split(',')]
    assert len(weights_list) == len(test_segments_list)
    if args.coeff is None:
        coeff_list = [1] * len(weights_list)
    else:
        coeff_list = [float(c) for c in args.coeff.split(',')]
    
    data_iter_list = []
    net_list = []
    modality_list = []
    total_num = None
    
    for this_weights, this_test_segments in zip(weights_list, test_segments_list):
        is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
        if 'combined' in this_weights:
            modality = 'combined'
        elif 'RGB' in this_weights:
            modality = 'RGB'
        elif 'mono' in this_weights:
            modality = 'mono'
        else:
            print("Modality not recognized!!")
            exit()
        num_class, args.train_list, _, root_path, prefix = \
            dataset_config.return_dataset(args.dataset, modality, args.text_file)
        print(f"=> shift: {is_shift}, shift_div: {shift_div}, shift_place: {shift_place}")
        if modality in ['mono', 'combined']:
            modality = 'RGB'
        modality_list.append(modality)
        this_arch = this_weights.split('TSM_')[1].split('_')[2]
        net = TSN(num_class, this_test_segments if is_shift else 1, modality,
                  base_model=this_arch,
                  consensus_type=args.crop_fusion_type,
                  img_feature_dim=args.img_feature_dim,
                  pretrain=args.pretrain,
                  is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
                  non_local=('_nl' in this_weights))
        if 'tpool' in this_weights:
            from ops.temporal_shift import make_temporal_pool
            make_temporal_pool(net.base_model, this_test_segments)
        print(f"🔹 Loading pretrained weights from: {this_weights}")
        checkpoint = torch.load(this_weights, map_location=lambda storage, loc: storage.cuda())
        checkpoint = checkpoint['state_dict']
        sd = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                        'base_model.classifier.bias': 'new_fc.bias'}
        for k, new_key in replace_dict.items():
            if k in sd:
                sd[new_key] = sd.pop(k)
        net.load_state_dict(sd)
        net = net.cuda()
        net.eval()
    
        input_size = net.scale_size if args.full_res else net.input_size
        if args.test_crops == 1:
            cropping = transforms.Compose([
                GroupScale(net.scale_size),
                GroupCenterCrop(input_size),
            ])
        elif args.test_crops == 3:
            cropping = transforms.Compose([
                GroupFullResSample(input_size, net.scale_size, flip=False)
            ])
        elif args.test_crops == 5:
            cropping = transforms.Compose([
                GroupOverSample(input_size, net.scale_size, flip=False)
            ])
        elif args.test_crops == 10:
            cropping = transforms.Compose([
                GroupOverSample(input_size, net.scale_size)
            ])
        else:
            raise ValueError(f"Only 1, 5, or 10 crops are supported, got {args.test_crops}")
        transform_pipeline = transforms.Compose([
            cropping,
            Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
            GroupNormalize(net.input_mean, net.input_std),
        ])
    
        # Use the command-line provided evaluation file (args.eval_list)
        eval_loader = DataLoader(
            TSNDataSet(root_path, args.eval_list, num_segments=this_test_segments,
                       new_length=1 if modality=="RGB" else 5,
                       modality=modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       transform=transform_pipeline),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
    
        print("\nSanity Check on Evaluation Set Predictions:")
        for i, (input_data, target, *rest) in enumerate(eval_loader):
            input_data = input_data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            print(f"Input Shape: {input_data.shape}")
            with torch.no_grad():
                output = net(input_data)
                pred = torch.argmax(output, dim=1)
            print(f"Batch {i} - True Labels: {target[:10].cpu().numpy()}")
            print(f"Batch {i} - Predicted Labels: {pred[:10].cpu().numpy()}")
            break
    
        net = torch.nn.DataParallel(net)
        net.eval()
        data_iter_list.append(enumerate(eval_loader))
        net_list.append(net)
    
    # ------------------------------
    # Create Calibration (Validation) DataLoader for CP
    # ------------------------------
    calib_loader = DataLoader(
        TSNDataSet(root_path, args.calib_list, num_segments=args.num_segments,
                   new_length=1, modality='RGB', image_tmpl=prefix,
                   transform=transform_pipeline, test_mode=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    
    print("\nSanity Check on Calibration (Validation) Set Predictions:")
    for i, (input_data, target, *rest) in enumerate(calib_loader):
        input_data = input_data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            output = net_list[0](input_data)
            pred = torch.argmax(output, dim=1)
        print(f"Batch {i} - True Labels: {target[:10].cpu().numpy()}")
        print(f"Batch {i} - Predicted Labels: {pred[:10].cpu().numpy()}")
        break
    
    # ------------------------------
    # Conformal Prediction
    # ------------------------------
    cp_model = net_list[0].module if isinstance(net_list[0], torch.nn.DataParallel) else net_list[0]
    if not hasattr(cp_model, 'predict_proba'):
        def predict_proba_wrapper(input_tensor):
            cp_model.eval()
            with torch.no_grad(), autocast():
                logits = cp_model(input_tensor)
                probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()
        cp_model.predict_proba = predict_proba_wrapper
    
    cp = SingleLabelConformalPredictor(cp_model, significance=args.significance, device=torch.device("cuda"))
    cp.calibrate(calib_loader)
    
    # Evaluate CP on the evaluation set and save CSV
    print("\n[CP] Evaluating on Evaluation Set...")
    cp_results_eval, top5_acc = cp.evaluate(eval_loader)
    csv_filename_eval = "cp_results_evaluation.csv"
    with open(csv_filename_eval, mode="w", newline="") as csvfile:
        fieldnames = list(cp_results_eval[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in cp_results_eval:
            writer.writerow(row)
    print(f"\nCP evaluation results saved to {csv_filename_eval}")
    
    # Evaluate CP on the calibration (validation) set and save CSV
    print("\n[CP] Evaluating on Calibration (Validation) Set...")
    cp_results_val, top5_acc_val = cp.evaluate(calib_loader)
    csv_filename_val = "cp_results_validation.csv"
    with open(csv_filename_val, mode="w", newline="") as csvfile:
        fieldnames = list(cp_results_val[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in cp_results_val:
            writer.writerow(row)
    print(f"\nCP validation results saved to {csv_filename_val}")
    
    # Compute Top-1 accuracy from the CP validation results
    val_true = np.array([res["true_label"] for res in cp_results_val])
    val_pred = np.array([res["predicted_label"] for res in cp_results_val])
    prec1 = np.mean(val_true == val_pred) * 100.0
    
    print("\nValidation Set Metrics:")
    print(f"Overall Prec@1: {prec1:.2f}%")
    print(f"Overall Prec@5: {top5_acc_val*100:.2f}%")
    print(f"Combined Average p-value (Evaluation): {combine_p_values_fisher([res['p_value'] for res in cp_results_eval]):.3f}")
    
    # Confusion Matrix for validation set
    cm = confusion_matrix(val_true, val_pred)
    print("\nConfusion Matrix (Validation Set):")
    print(cm)
    
    video_p_values = [res["p_value"] for res in cp_results_eval]
    combined_video_p_value = combine_p_values_fisher(video_p_values)
    print("\nCombined video p-value =", combined_video_p_value)
    overall_status = "OOD" if combined_video_p_value < args.significance else "in-distribution"
    print(f"\nOverall: Video is detected as {overall_status}")
    
if __name__ == '__main__':
    main()
