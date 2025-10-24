"""
Dataset validation utilities for checking label and image integrity.
"""
from pathlib import Path


class DatasetValidator:
    """Validates dataset for common issues."""
    
    def __init__(self, dataset_root, num_classes):
        """
        Initialize validator.
        
        Args:
            dataset_root: Root directory of dataset
            num_classes: Number of classes in dataset
        """
        self.dataset_root = Path(dataset_root)
        self.num_classes = num_classes
    
    def validate_label_file(self, label_path):
        """
        Validate a single label file.
        
        Args:
            label_path: Path to label file
            
        Returns:
            list: List of issue strings (empty if valid)
        """
        issues = []
        
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    issues.append(f"Line {line_num}: Not enough values ({len(parts)}/5)")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check class ID
                    if class_id < 0 or class_id >= self.num_classes:
                        issues.append(f"Line {line_num}: Invalid class ID {class_id}")
                    
                    # Check coordinates
                    if not (0 <= x_center <= 1):
                        issues.append(f"Line {line_num}: x_center {x_center} out of range")
                    if not (0 <= y_center <= 1):
                        issues.append(f"Line {line_num}: y_center {y_center} out of range")
                    if not (0 < width <= 1):
                        issues.append(f"Line {line_num}: width {width} out of range")
                    if not (0 < height <= 1):
                        issues.append(f"Line {line_num}: height {height} out of range")
                    
                except ValueError as e:
                    issues.append(f"Line {line_num}: Parse error - {e}")
        
        return issues
    
    def validate_split(self, split='train'):
        """
        Validate a dataset split (train or test).
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            dict: Validation results
        """
        labels_dir = self.dataset_root / 'labels' / split
        images_dir = self.dataset_root / 'images' / split
        
        results = {
            'split': split,
            'label_files': 0,
            'image_files': 0,
            'missing_labels': [],
            'missing_images': [],
            'invalid_labels': {},
            'total_issues': 0
        }
        
        if not labels_dir.exists() or not images_dir.exists():
            results['error'] = "Directories not found"
            return results
        
        # Get files
        label_files = list(labels_dir.glob('*.txt'))
        image_files = list(images_dir.glob('*.jpg'))
        
        results['label_files'] = len(label_files)
        results['image_files'] = len(image_files)
        
        # Check for missing pairs
        label_stems = {f.stem for f in label_files}
        image_stems = {f.stem for f in image_files}
        
        results['missing_labels'] = list(image_stems - label_stems)
        results['missing_images'] = list(label_stems - image_stems)
        
        # Validate label content
        for label_file in label_files:
            issues = self.validate_label_file(label_file)
            if issues:
                results['invalid_labels'][label_file.name] = issues
                results['total_issues'] += len(issues)
        
        return results
    
    def validate_dataset(self):
        """
        Validate entire dataset (train and test).
        
        Returns:
            dict: Complete validation results
        """
        results = {
            'train': self.validate_split('train'),
            'test': self.validate_split('test')
        }
        
        # Add summary
        total_issues = (
            results['train']['total_issues'] + 
            results['test']['total_issues']
        )
        
        results['summary'] = {
            'is_valid': total_issues == 0,
            'total_issues': total_issues
        }
        
        return results
    
    def print_report(self, results):
        """
        Print a formatted validation report.
        
        Args:
            results: Results from validate_dataset()
        """
        print("\n" + "="*60)
        print("Dataset Validation Report")
        print("="*60)
        
        for split in ['train', 'test']:
            split_results = results[split]
            
            print(f"\n{split.upper()} Set:")
            print(f"  Label files: {split_results['label_files']}")
            print(f"  Image files: {split_results['image_files']}")
            
            if split_results['missing_labels']:
                print(f"  Missing labels: {len(split_results['missing_labels'])}")
            
            if split_results['missing_images']:
                print(f"  Missing images: {len(split_results['missing_images'])}")
            
            if split_results['invalid_labels']:
                print(f"  Invalid label files: {len(split_results['invalid_labels'])}")
                print(f"  Total issues: {split_results['total_issues']}")
                
                # Show first few files with issues
                for i, (filename, issues) in enumerate(split_results['invalid_labels'].items()):
                    if i >= 3:  # Show only first 3
                        break
                    print(f"\n  {filename}:")
                    for issue in issues[:3]:  # Show only first 3 issues per file
                        print(f"    - {issue}")
            else:
                print("  ✓ All labels valid")
        
        print("\n" + "="*60)
        if results['summary']['is_valid']:
            print("✓ Dataset is valid!")
        else:
            print(f"⚠ Found {results['summary']['total_issues']} issues")
        print("="*60)

