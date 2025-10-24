"""
Evaluation script for logo detection model.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics.utils.metrics import bbox_iou
from .base import YOLOModel


def log_misclassifications(validator):
    """
    Logs misclassifications during validation to the console.
    This function is intended to be used as a callback with the YOLO model.
    """
    # Use the same thresholds as the validator
    iou_threshold = validator.args.iou
    conf_threshold = validator.args.conf
    
    # Check if there are any predictions to process
    if not hasattr(validator, 'preds') or validator.preds is None:
        return
        
    preds = validator.preds
    batch = validator.batch
    class_names = validator.names

    # Group ground truth labels by their image index in the batch
    gt_labels = []
    if 'batch_idx' in batch and 'cls' in batch and 'bboxes' in batch:
        for i in range(len(validator.im_files)):
            mask = batch['batch_idx'] == i
            cls = batch['cls'][mask].squeeze(1) if batch['cls'][mask].numel() > 0 else torch.tensor([])
            bboxes = batch['bboxes'][mask]
            
            # Ensure tensors are 2D before concatenation
            if cls.dim() == 1:
                cls = cls.unsqueeze(1)
            
            if cls.numel() > 0:
                gt_labels.append(torch.cat((cls, bboxes), 1))
            else:
                gt_labels.append(torch.empty(0, 5))
    else:
        # If batch structure is different, exit to avoid errors
        return

    # Compare predictions with ground truth for each image
    for i, pred in enumerate(preds):
        if pred.shape[0] == 0 or i >= len(gt_labels) or gt_labels[i].shape[0] == 0:
            continue

        # Filter predictions by confidence threshold
        pred = pred[pred[:, 4] > conf_threshold]
        if pred.shape[0] == 0:
            continue
            
        gt = gt_labels[i].to(pred.device)
        image_path = validator.im_files[i]

        # Un-normalize GT boxes from xywh (normalized) to xyxy (pixels)
        h, w = batch['img'][i].shape[1:]
        gt_bboxes_normalized = gt[:, 1:]
        gt_bboxes_unnormalized_xywh = gt_bboxes_normalized.clone()
        gt_bboxes_unnormalized_xywh[:, 0] *= w
        gt_bboxes_unnormalized_xywh[:, 1] *= h
        gt_bboxes_unnormalized_xywh[:, 2] *= w
        gt_bboxes_unnormalized_xywh[:, 3] *= h
        
        # Convert to xyxy
        gt_xyxy = torch.empty_like(gt_bboxes_unnormalized_xywh)
        gt_xyxy[:, 0] = gt_bboxes_unnormalized_xywh[:, 0] - gt_bboxes_unnormalized_xywh[:, 2] / 2
        gt_xyxy[:, 1] = gt_bboxes_unnormalized_xywh[:, 1] - gt_bboxes_unnormalized_xywh[:, 3] / 2
        gt_xyxy[:, 2] = gt_bboxes_unnormalized_xywh[:, 0] + gt_bboxes_unnormalized_xywh[:, 2] / 2
        gt_xyxy[:, 3] = gt_bboxes_unnormalized_xywh[:, 1] + gt_bboxes_unnormalized_xywh[:, 3] / 2

        # Calculate IoU between predictions and ground truth
        ious = bbox_iou(pred[:, :4], gt_xyxy, xywh=False)

        if ious.shape[1] == 0:
            continue
            
        max_ious, argmax_ious = torch.max(ious, dim=1)

        for pred_idx, gt_idx in enumerate(argmax_ious):
            if max_ious[pred_idx] > iou_threshold:
                pred_class_idx = int(pred[pred_idx, 5])
                gt_class_idx = int(gt[gt_idx, 0])

                if pred_class_idx != gt_class_idx:
                    pred_class_name = class_names[pred_class_idx]
                    gt_class_name = class_names[gt_class_idx]
                    print(f"Misclassification in {image_path}: Detected '{pred_class_name}' but should be '{gt_class_name}'.")


class LogoEvaluator(YOLOModel):
    """Evaluator for logo detection model."""
    
    def parse_args(self):
        """Parse evaluation arguments."""
        parser = argparse.ArgumentParser(description='Evaluate YOLOv8 model for logo detection')
        parser.add_argument('--weights', type=str, required=True,
                          help='path to trained weights')
        parser.add_argument('--data', type=str, default='configs/data.yaml',
                          help='path to data.yaml file')
        parser.add_argument('--iou-thres', type=float, default=0.5,
                          help='IOU threshold for evaluation')
        parser.add_argument('--conf-thres', type=float, default=0.25,
                          help='confidence threshold for evaluation')
        parser.add_argument('--batch-size', type=int, default=16,
                          help='batch size for evaluation')
        parser.add_argument('--img-size', type=int, default=640,
                          help='input image size')
        parser.add_argument('--save-json', action='store_true',
                          help='save results to JSON file')
        parser.add_argument('--output-dir', type=str, default='results/evaluation',
                          help='output directory for results')
        return parser.parse_args()
    
    def run(self, args):
        """Execute evaluation."""
        # Load trained model
        self.load_model(args.weights)
        
        # Print evaluation configuration
        self._print_config(args)
        
        # Add callback to log misclassifications
        self.model.add_callback("on_val_batch_end", log_misclassifications)
        
        # Run evaluation
        results = self.model.val(
            data=args.data,
            batch=args.batch_size,
            imgsz=args.img_size,
            conf=args.conf_thres,
            iou=args.iou_thres,
            save_json=args.save_json,
            project=args.output_dir,
            name='val'
        )
        
        # Remove callback after evaluation
        self.model.clear_callback("on_val_batch_end")
        
        # Save and print results
        output_path = Path(results.save_dir)
        metrics = self._save_results(results, output_path, args.save_json)
        self._print_results(metrics)

        # Save metrics as figures
        self._save_metrics_figures(results, output_path)
        
        return results
    
    def _print_config(self, args):
        """Print evaluation configuration."""
        print("\n" + "="*60)
        print("Evaluation Configuration")
        print("="*60)
        print(f"  Model: {args.weights}")
        print(f"  Data: {args.data}")
        print(f"  IOU threshold: {args.iou_thres}")
        print(f"  Confidence threshold: {args.conf_thres}")
        print("="*60 + "\n")
    
    def _save_results(self, results, output_dir, save_json):
        """Save evaluation results."""
        metrics = {
            'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
            'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
            'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
            'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0))
        }
        
        if save_json:
            output_path = output_dir / 'metrics.json'
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved to: {output_path}")
        
        return metrics
    
    def _save_metrics_figures(self, results, output_dir: Path):
        """Save per-class and summary metrics as PNG images."""
        # Per-class metrics
        class_metrics_data = []
        for i in range(results.box.nc):
            class_metrics_data.append([
                results.names[i],
                results.box.p[i],
                results.box.r[i],
                results.box.ap50[i],
                results.box.ap[i],
            ])

        if class_metrics_data:
            class_metrics_df = pd.DataFrame(
                class_metrics_data,
                columns=['Class', 'Precision', 'Recall', 'mAP@.50', 'mAP@.50-.95']
            )
            class_metrics_df[['Precision', 'Recall', 'mAP@.50', 'mAP@.50-.95']] = \
                class_metrics_df[['Precision', 'Recall', 'mAP@.50', 'mAP@.50-.95']].round(4)

            fig, ax = plt.subplots(figsize=(12, (len(class_metrics_df) + 1) * 0.5))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=class_metrics_df.values, colLabels=class_metrics_df.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            plt.title('Per-Class Metrics', fontsize=16, pad=20)
            fig.tight_layout()
            per_class_path = output_dir / "per_class_metrics.png"
            fig.savefig(per_class_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Per-class metrics figure saved to: {per_class_path}")

        # Summary metrics
        summary_metrics = {
            'Precision': f"{results.box.mp:.4f}",
            'Recall': f"{results.box.mr:.4f}",
            'mAP50': f"{results.box.map50:.4f}",
            'mAP50-95': f"{results.box.map:.4f}",
        }
        summary_df = pd.DataFrame(list(summary_metrics.items()), columns=['Metric', 'Value'])
        
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title('Summary Metrics', fontsize=16, pad=20)
        fig.tight_layout()
        summary_path = output_dir / "summary_metrics.png"
        fig.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Summary metrics figure saved to: {summary_path}")

    def _print_results(self, metrics):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  mAP50:     {metrics['mAP50']:.4f}")
        print(f"  mAP50-95:  {metrics['mAP50-95']:.4f}")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    evaluator = LogoEvaluator()
    evaluator.execute()


if __name__ == "__main__":
    main()

