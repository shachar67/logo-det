"""
Base class for YOLO model operations (train, evaluate, predict).
Provides common functionality and structure.
"""
import os
from pathlib import Path
from abc import ABC, abstractmethod
from ultralytics import YOLO, settings


class YOLOModel(ABC):
    """Base class for YOLO model operations."""
    
    def __init__(self):
        """Initialize the script with common settings."""
        # Disable wandb
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_DISABLED'] = 'true'
        settings.update({'wandb': False})
        
        # Setup directories
        self.weights_dir = Path('weights')
        self.weights_dir.mkdir(exist_ok=True)
        
        self.model = None
    
    def load_model(self, weights_path):
        """
        Load a YOLO model from weights.
        
        Args:
            weights_path: Path to model weights
            
        Returns:
            YOLO: Loaded model
        """
        weights_path = Path(weights_path)
        
        # Check if weights exist
        if not weights_path.exists():
            # If only filename, check in weights directory
            if weights_path.parent == Path('.'):
                weights_in_dir = self.weights_dir / weights_path.name
                if weights_in_dir.exists():
                    weights_path = weights_in_dir
                else:
                    print(f"Weights not found at {weights_path}")
                    print(f"YOLO will download pretrained weights to: {self.weights_dir}")
                    weights_path = self.weights_dir / weights_path.name
        
        print(f"Loading model from: {weights_path}")
        self.model = YOLO(str(weights_path))
        return self.model
    
    @abstractmethod
    def parse_args(self):
        """Parse command-line arguments. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run(self, args):
        """Run the script logic. Must be implemented by subclasses."""
        pass
    
    def execute(self):
        """Main execution method."""
        args = self.parse_args()
        return self.run(args)

