"""
Class scanner utilities for detecting classes in label files.
"""
from pathlib import Path


class ClassScanner:
    """Scans label files to detect unique classes."""
    
    def __init__(self, labels_dir):
        """
        Initialize scanner.
        
        Args:
            labels_dir: Directory containing label files
        """
        self.labels_dir = Path(labels_dir)
    
    def scan(self):
        """
        Scan all label files and extract unique class names.
        
        Returns:
            list: Sorted list of unique class names
        """
        classes = set()
        label_files = list(self.labels_dir.glob('*.txt'))
        
        print(f"Scanning {len(label_files)} label files...")
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_name = parts[0].lower()
                            classes.add(class_name)
        
        return sorted(list(classes))
    
    def create_class_mapping(self):
        """
        Scan classes and create index mapping.
        
        Returns:
            dict: Mapping from class indices to names, e.g., {0: 'nike', 1: 'adidas'}
        """
        classes = self.scan()
        id_to_class = {idx: class_name for idx, class_name in enumerate(classes)}
        
        print(f"\nFound {len(classes)} classes:")
        for idx, class_name in id_to_class.items():
            print(f"  {idx}: {class_name}")
        
        return id_to_class
    
    def create_reverse_mapping(self):
        """
        Create reverse mapping (name to index).
        
        Returns:
            dict: Mapping from class names to indices, e.g., {'nike': 0, 'adidas': 1}
        """
        id_to_class = self.create_class_mapping()
        return {v: k for k, v in id_to_class.items()}

