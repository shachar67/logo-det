"""
Label conversion utilities for converting class names to numeric indices.
"""
from pathlib import Path
import shutil


class LabelConverter:
    """Converts label files from class names to numeric indices."""
    
    def __init__(self, class_mapping):
        """
        Initialize converter with class mapping.
        
        Args:
            class_mapping: Dict mapping class names to indices, e.g., {'nike': 0, 'adidas': 1}
        """
        self.class_mapping = class_mapping
    
    def is_converted(self, label_path):
        """
        Check if a label file is already converted.
        
        Args:
            label_path: Path to label file
            
        Returns:
            bool: True if already converted (starts with digit)
        """
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line and first_line[0].isdigit():
                return True
        return False
    
    def convert_file(self, label_path, backup=True):
        """
        Convert a single label file from class names to indices.
        
        Args:
            label_path: Path to label file
            backup: Whether to create a backup (.txt.bak)
            
        Returns:
            bool: True if converted, False if already converted
        """
        # Check if already converted
        if self.is_converted(label_path):
            return False
        
        # Read original file
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Create backup
        if backup:
            backup_path = label_path.with_suffix('.txt.bak')
            if not backup_path.exists():
                shutil.copy(label_path, backup_path)
        
        # Convert lines
        converted_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                class_name = parts[0].lower()
                coords = parts[1:5]
                
                # Convert class name to index
                if class_name in self.class_mapping:
                    class_id = self.class_mapping[class_name]
                    new_line = f"{class_id} {' '.join(coords)}\n"
                    converted_lines.append(new_line)
                else:
                    print(f"Warning: Unknown class '{class_name}' in {label_path.name}")
        
        # Write converted labels
        with open(label_path, 'w') as f:
            f.writelines(converted_lines)
        
        return True
    
    def convert_directory(self, labels_dir):
        """
        Convert all label files in a directory.
        
        Args:
            labels_dir: Path to directory containing label files
            
        Returns:
            int: Number of files converted
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            print(f"Warning: Directory not found: {labels_dir}")
            return 0
        
        # Skip .cache files
        label_files = [f for f in labels_dir.glob('*.txt') if not f.name.endswith('.cache')]
        converted_count = 0
        
        for label_file in label_files:
            if self.convert_file(label_file, backup=True):
                converted_count += 1
        
        return converted_count
    
    def convert_dataset(self, dataset_root):
        """
        Convert all labels in train and test directories.
        
        Args:
            dataset_root: Root directory of dataset
            
        Returns:
            dict: Statistics about conversion
        """
        dataset_root = Path(dataset_root)
        
        results = {
            'train': {'converted': 0, 'total': 0},
            'test': {'converted': 0, 'total': 0}
        }
        
        for split in ['train', 'test']:
            labels_dir = dataset_root / 'labels' / split
            
            if labels_dir.exists():
                # Skip .cache files when counting
                total_files = len([f for f in labels_dir.glob('*.txt') if not f.name.endswith('.cache')])
                converted = self.convert_directory(labels_dir)
                
                results[split]['total'] = total_files
                results[split]['converted'] = converted
        
        return results

