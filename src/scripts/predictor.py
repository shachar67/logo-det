"""
Prediction/inference script for logo detection.
"""
import argparse
from pathlib import Path
from .base import YOLOModel


class LogoPredictor(YOLOModel):
    """Predictor for logo detection model."""
    
    def parse_args(self):
        """Parse prediction arguments."""
        parser = argparse.ArgumentParser(description='Run inference with trained YOLOv8 model')
        parser.add_argument('--weights', type=str, required=True,
                          help='path to trained weights')
        parser.add_argument('--source', type=str, required=True,
                          help='path to image, video, or directory')
        parser.add_argument('--conf-thres', type=float, default=0.25,
                          help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45,
                          help='NMS IOU threshold')
        parser.add_argument('--img-size', type=int, default=640,
                          help='input image size')
        parser.add_argument('--output-dir', type=str, default='results/predictions',
                          help='output directory for predictions')
        parser.add_argument('--save', action='store_true',
                          help='save images with predictions')
        parser.add_argument('--show', action='store_true',
                          help='show results')
        parser.add_argument('--save-txt', action='store_true',
                          help='save results to txt files')
        parser.add_argument('--save-conf', action='store_true',
                          help='save confidences in txt files')
        return parser.parse_args()
    
    def run(self, args):
        """Execute prediction."""
        # Load trained model
        self.load_model(args.weights)
        
        # Print prediction configuration
        self._print_config(args)
        
        # Run prediction
        results = self.model.predict(
            source=args.source,
            imgsz=args.img_size,
            conf=args.conf_thres,
            iou=args.iou_thres,
            save=args.save,
            show=args.show,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            project=args.output_dir,
            name='predict',
            exist_ok=True
        )
        
        # Print results summary
        self._print_results(results, args)
        
        return results
    
    def _print_config(self, args):
        """Print prediction configuration."""
        print("\n" + "="*60)
        print("Prediction Configuration")
        print("="*60)
        print(f"  Model: {args.weights}")
        print(f"  Source: {args.source}")
        print(f"  Confidence threshold: {args.conf_thres}")
        print(f"  IOU threshold: {args.iou_thres}")
        print(f"  Output directory: {args.output_dir}")
        print("="*60 + "\n")
    
    def _print_results(self, results, args):
        """Print prediction results."""
        print("\n" + "="*60)
        print("Prediction Completed!")
        print("="*60)
        
        # Count detections
        total_detections = 0
        for result in results:
            if result.boxes is not None:
                num_boxes = len(result.boxes)
                total_detections += num_boxes
                
                # Print detections for this image
                if num_boxes > 0:
                    print(f"\n{result.path}:")
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        print(f"  - {class_name}: {conf:.2f}")
        
        print(f"\nTotal detections: {total_detections}")
        
        if args.save:
            print(f"Results saved to: {args.output_dir}/predict")
        
        print("="*60 + "\n")


def main():
    """Main entry point."""
    predictor = LogoPredictor()
    predictor.execute()


if __name__ == "__main__":
    main()

