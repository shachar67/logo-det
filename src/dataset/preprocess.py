"""
Main data preprocessing script.
Orchestrates dataset verification, class detection, and label conversion.
"""
from pathlib import Path
import yaml
import random
import shutil

from .class_scanner import ClassScanner
from .label_converter import LabelConverter
from .dataset_validator import DatasetValidator


class DatasetPreprocessor:
    """Main class for preprocessing the logo detection dataset."""
    
    def __init__(self, dataset_root='data/logos-dataset-section-1/logos_dataset'):
        """
        Initialize dataset preprocessor.
        
        Args:
            dataset_root: Path to dataset root directory
        """
        self.dataset_root = Path(dataset_root)
        self.config_dir = Path('configs')
        self.data_yaml_path = self.config_dir / 'data.yaml'
    
    def create_validation_split(self, split_ratio: float = 0.2):
        """
        Splits the training data into training and validation sets if the
        validation set does not exist.
        """
        print("\n[Step 1/5] Checking for validation split...")
        
        train_images_dir = self.dataset_root / 'images' / 'train'
        val_images_dir = self.dataset_root / 'images' / 'val'
        
        if val_images_dir.exists() and any(val_images_dir.iterdir()):
            print("  ✓ Validation set already exists. Skipping.")
            return

        print("  ! Validation set not found. Creating split...")
        train_labels_dir = self.dataset_root / 'labels' / 'train'
        val_labels_dir = self.dataset_root / 'labels' / 'val'

        # Create validation directories
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        # Get image files and shuffle
        image_files = sorted([f for f in train_images_dir.glob('*.jpg')])
        random.seed(42)
        random.shuffle(image_files)

        # Move files
        num_val_files = int(len(image_files) * split_ratio)
        val_files = image_files[:num_val_files]

        for img_path in val_files:
            label_path = train_labels_dir / (img_path.stem + '.txt')
            shutil.move(str(img_path), str(val_images_dir / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(val_labels_dir / label_path.name))

        print(f"  ✓ Moved {len(val_files)} files to validation set.")

    def verify_structure(self):
        """Verify that required directories exist."""
        print("\n[Step 2/5] Verifying dataset structure...")
        
        required_dirs = [
            self.dataset_root / 'images' / 'train',
            self.dataset_root / 'images' / 'test',
            self.dataset_root / 'labels' / 'train',
            self.dataset_root / 'labels' / 'test'
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
            
            num_files = len(list(dir_path.glob('*.*')))
            print(f"  ✓ {dir_path.relative_to('.')}: {num_files} files")
    
    def detect_classes(self):
        """Detect all classes in the dataset."""
        print("\n[Step 3/5] Detecting classes...")
        
        train_labels_dir = self.dataset_root / 'labels' / 'train'
        scanner = ClassScanner(train_labels_dir)
        
        # Get class mappings
        id_to_class = scanner.create_class_mapping()
        name_to_id = scanner.create_reverse_mapping()
        
        return id_to_class, name_to_id
    
    def create_data_config(self, id_to_class):
        """Create data.yaml configuration file."""
        print("\n[Step 4/5] Creating data.yaml...")
        
        data = {
            'path': str(self.dataset_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': id_to_class,
            'nc': len(id_to_class)
        }
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Save data.yaml
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ Created: {self.data_yaml_path}")
        
        return data
    
    def convert_labels(self, name_to_id):
        """Convert label files from class names to numeric indices."""
        print("\n[Step 5/5] Converting labels to numeric indices...")
        
        converter = LabelConverter(name_to_id)
        results = converter.convert_dataset(self.dataset_root)
        
        # Print results
        for split in ['train', 'test']:
            total = results[split]['total']
            converted = results[split]['converted']
            skipped = total - converted
            
            if converted > 0:
                print(f"  {split}: Converted {converted}/{total} files (skipped {skipped} already converted)")
            else:
                print(f"  {split}: All {total} files already converted")
        
        total_converted = results['train']['converted'] + results['test']['converted']
        return total_converted
    
    def validate(self, num_classes):
        """Validate the prepared dataset."""
        print("\n[Optional] Validating dataset...")
        
        validator = DatasetValidator(self.dataset_root, num_classes)
        results = validator.validate_dataset()
        
        if results['summary']['is_valid']:
            print("  ✓ Dataset validation passed!")
            return True
        else:
            print(f"  ⚠ Found {results['summary']['total_issues']} validation issues")
            validator.print_report(results)
            return False
    
    def run(self, validate=True):
        """
        Run the complete preparation pipeline.
        
        Args:
            validate: Whether to run validation after preparation
        """
        print("=" * 60)
        print("Logo Detection Dataset Preprocessing")
        print("=" * 60)
        
        try:
            # Step 1: Create validation split if needed
            self.create_validation_split()

            # Step 2: Verify structure
            self.verify_structure()
            
            # Step 3: Detect classes
            id_to_class, name_to_id = self.detect_classes()
            
            # Step 4: Create data.yaml
            data_config = self.create_data_config(id_to_class)
            
            # Step 5: Convert labels
            converted_count = self.convert_labels(name_to_id)
            
            # Optional: Validate
            if validate:
                self.validate(data_config['nc'])
            
            # Print summary
            print("\n" + "=" * 60)
            print("✓ Dataset Preprocessing Complete!")
            print("=" * 60)
            print("\nSummary:")
            print(f"  Classes: {data_config['nc']}")
            print(f"  Data config: {self.data_yaml_path}")
            print(f"  Labels converted: {converted_count}")
            print("\nYou can now train the model with:")
            print("  python train.py")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Error during preprocessing: {e}")
            raise


def main():
    """Main entry point."""
    preprocessor = DatasetPreprocessor()
    preprocessor.run(validate=True)


if __name__ == "__main__":
    main()

