"""
Training script for logo detection using YOLOv8.
"""
import argparse
from .base import YOLOModel


class LogoTrainer(YOLOModel):
    """Trainer for logo detection model."""
    
    def parse_args(self):
        """Parse training arguments."""
        parser = argparse.ArgumentParser(description='Train YOLOv8 model for logo detection')
        parser.add_argument('--data', type=str, default='configs/data.yaml',
                          help='path to data.yaml file')
        parser.add_argument('--epochs', type=int, default=50,
                          help='number of epochs to train for')
        parser.add_argument('--batch-size', type=int, default=8,
                          help='batch size for training')
        parser.add_argument('--img-size', type=int, default=640,
                          help='input image size')
        parser.add_argument('--weights', type=str, default='weights/yolov8n.pt',
                          help='path to initial weights')
        parser.add_argument('--lr', type=float, default=0.001,
                          help='initial learning rate')
        parser.add_argument('--patience', type=int, default=10,
                          help='epochs to wait for no improvement')
        parser.add_argument('--project', type=str, default='logo_detection',
                          help='project name')
        parser.add_argument('--name', type=str, default='train',
                          help='experiment name')
        parser.add_argument('--device', type=str, default='',
                          help='device to train on (e.g., 0 or cpu)')
        return parser.parse_args()
    
    def run(self, args):
        """Execute training."""
        # Load model
        self.load_model(args.weights)
        
        # Print training configuration
        self._print_config(args)
        
        # Train the model
        results = self.model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size,
            lr0=args.lr,
            patience=args.patience,
            save=True,
            project=args.project,
            name=args.name,
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            verbose=True,
            amp=False,  # Disable mixed precision to avoid NaN issues
            device=args.device if args.device else None,
            # Conservative augmentations
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
        )
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _print_config(self, args):
        """Print training configuration."""
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"  Data: {args.data}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Image size: {args.img_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Project: {args.project}")
        print(f"  Name: {args.name}")
        print("="*60 + "\n")
    
    def _print_results(self, results):
        """Print training results."""
        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
        print(f"Results saved to: {results.save_dir}")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            print("\nFinal Metrics:")
            metrics = results.results_dict
            print(f"  mAP50:     {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"  Recall:    {metrics.get('metrics/recall(B)', 'N/A')}")
        
        print("\nBest model saved at:")
        print(f"  {results.save_dir}/weights/best.pt")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    trainer = LogoTrainer()
    trainer.execute()


if __name__ == "__main__":
    main()

