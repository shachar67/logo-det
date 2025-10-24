#!/usr/bin/env python3
"""
Main entry point for logo detection project.
Provides a unified CLI interface for all operations.

Usage:
    python main.py train --epochs 50 --batch-size 16
    python main.py evaluate --weights best.pt
    python main.py predict --weights best.pt --source image.jpg
"""
import argparse
from src.scripts.trainer import LogoTrainer
from src.scripts.evaluator import LogoEvaluator
from src.scripts.predictor import LogoPredictor
from src.scripts.zero_shot import ZeroShotPredictor, API_KEY


def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='logo-detect',
        description='Logo Detection System using YOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --epochs 50 --batch-size 16
  
  # Evaluate a trained model
  python main.py evaluate --weights logo_detection/train/weights/best.pt
  
  # Run predictions
  python main.py predict --weights best.pt --source image.jpg --save
  
  # Preprocess dataset
  python src/dataset/preprocess.py
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command',
        required=True,
        help='Command to execute'
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train a logo detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument('--data', type=str, default='configs/data.yaml',
                             help='path to data.yaml file')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=8,
                             help='batch size')
    train_parser.add_argument('--img-size', type=int, default=640,
                             help='input image size')
    train_parser.add_argument('--weights', type=str, default='weights/yolov8n.pt',
                             help='initial weights path')
    train_parser.add_argument('--lr', type=float, default=0.01,
                             help='learning rate')
    train_parser.add_argument('--patience', type=int, default=10,
                             help='early stopping patience')
    train_parser.add_argument('--project', type=str, default='logo_detection',
                             help='project name')
    train_parser.add_argument('--name', type=str, default='train',
                             help='experiment name')
    train_parser.add_argument('--device', type=str, default='',
                             help='device (e.g., 0 or cpu)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    eval_parser.add_argument('--weights', type=str, required=True,
                            help='path to trained weights')
    eval_parser.add_argument('--data', type=str, default='configs/data.yaml',
                            help='path to data.yaml file')
    eval_parser.add_argument('--iou-thres', type=float, default=0.5,
                            help='IOU threshold')
    eval_parser.add_argument('--conf-thres', type=float, default=0.25,
                            help='confidence threshold')
    eval_parser.add_argument('--batch-size', type=int, default=16,
                            help='batch size')
    eval_parser.add_argument('--img-size', type=int, default=640,
                            help='input image size')
    eval_parser.add_argument('--save-json', action='store_true',
                            help='save results to JSON')
    eval_parser.add_argument('--output-dir', type=str, default='results/evaluation',
                            help='output directory')
    
    # Predict command
    predict_parser = subparsers.add_parser(
        'predict',
        help='Run predictions on images/videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    predict_parser.add_argument('--weights', type=str, required=True,
                               help='path to trained weights')
    predict_parser.add_argument('--source', type=str, required=True,
                               help='image, video, or directory path')
    predict_parser.add_argument('--conf-thres', type=float, default=0.25,
                               help='confidence threshold')
    predict_parser.add_argument('--iou-thres', type=float, default=0.45,
                               help='NMS IOU threshold')
    predict_parser.add_argument('--img-size', type=int, default=640,
                               help='input image size')
    predict_parser.add_argument('--output-dir', type=str, default='results/predictions',
                               help='output directory')
    predict_parser.add_argument('--save', action='store_true',
                               help='save images with predictions')
    predict_parser.add_argument('--show', action='store_true',
                               help='show results')
    predict_parser.add_argument('--save-txt', action='store_true',
                               help='save results to txt')
    predict_parser.add_argument('--save-conf', action='store_true',
                               help='save confidences in txt')
    
    # Any Logo (Zero-Shot) command
    anylogo_parser = subparsers.add_parser(
        'any_logo',
        help='Run zero-shot prediction for any logo using a generative model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    anylogo_parser.add_argument('--source', type=str, required=True,
                              help='path to the image file')
    anylogo_parser.add_argument('--labels', type=str, default=None,
                              help='optional path to the ground truth labels file for evaluation')

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Route to appropriate command
    if args.command == 'train':
        trainer = LogoTrainer()
        # Convert args namespace to the format expected by trainer
        trainer_args = argparse.Namespace(
            data=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            weights=args.weights,
            lr=args.lr,
            patience=args.patience,
            project=args.project,
            name=args.name,
            device=args.device
        )
        trainer.run(trainer_args)
        
    elif args.command == 'evaluate':
        evaluator = LogoEvaluator()
        eval_args = argparse.Namespace(
            weights=args.weights,
            data=args.data,
            iou_thres=args.iou_thres,
            conf_thres=args.conf_thres,
            batch_size=args.batch_size,
            img_size=args.img_size,
            save_json=args.save_json,
            output_dir=args.output_dir
        )
        evaluator.run(eval_args)
        
    elif args.command == 'predict':
        predictor = LogoPredictor()
        predict_args = argparse.Namespace(
            weights=args.weights,
            source=args.source,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            img_size=args.img_size,
            output_dir=args.output_dir,
            save=args.save,
            show=args.show,
            save_txt=args.save_txt,
            save_conf=args.save_conf
        )
        predictor.run(predict_args)

    elif args.command == 'any_logo':
        predictor = ZeroShotPredictor(api_key=API_KEY)
        predictor.run(image_path=args.source, label_path=args.labels)


if __name__ == "__main__":
    main()

