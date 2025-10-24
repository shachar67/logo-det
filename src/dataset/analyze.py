"""
Dataset analysis and visualization script.
Generates statistics and plots about the dataset.
"""
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from tqdm import tqdm

def analyze_image_sizes(image_dir):
    """Analyze image dimensions and aspect ratios in the dataset."""
    sizes = []
    aspect_ratios = []
    print(f"\nAnalyzing images in {image_dir}...")
    
    image_paths = list(Path(image_dir).glob('*.jpg'))
    for img_path in tqdm(image_paths, desc="Processing images"):
        img = cv2.imread(str(img_path))
        if img is not None:
            height, width = img.shape[:2]
            sizes.append((width, height))
            aspect_ratios.append(width / height)
    
    return {
        'sizes': sizes,
        'aspect_ratios': aspect_ratios,
        'mean_width': np.mean([s[0] for s in sizes]),
        'mean_height': np.mean([s[1] for s in sizes]),
        'mean_aspect_ratio': np.mean(aspect_ratios)
    }

def analyze_labels(label_dir):
    """Analyze label distribution and bounding box properties."""
    class_counts = defaultdict(int)
    box_sizes = []
    box_aspect_ratios = []
    boxes_per_image = defaultdict(int)
    
    print(f"\nAnalyzing labels in {label_dir}...")
    label_paths = list(Path(label_dir).glob('*.txt'))
    
    for label_path in tqdm(label_paths, desc="Processing labels"):
        num_boxes = 0
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_name = parts[0]
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    class_counts[class_name] += 1
                    box_sizes.append((width, height))
                    box_aspect_ratios.append(width / height)
                    num_boxes += 1
        
        boxes_per_image[label_path.stem] = num_boxes
    
    return {
        'class_distribution': dict(class_counts),
        'box_sizes': box_sizes,
        'box_aspect_ratios': box_aspect_ratios,
        'boxes_per_image': dict(boxes_per_image),
        'mean_boxes_per_image': np.mean(list(boxes_per_image.values())),
        'max_boxes_per_image': max(boxes_per_image.values())
    }

def plot_statistics(train_stats, test_stats, output_dir='results/analysis'):
    """Create detailed visualizations of dataset statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')  # Using default matplotlib style
    
    # 1. Class Distribution
    plt.figure(figsize=(12, 6))
    train_classes = train_stats['label_stats']['class_distribution']
    test_classes = test_stats['label_stats']['class_distribution']
    
    x = np.arange(len(train_classes))
    width = 0.35
    
    plt.bar(x - width/2, list(train_classes.values()), width, label='Train', color='#2196F3', alpha=0.7)
    plt.bar(x + width/2, list(test_classes.values()), width, label='Test', color='#4CAF50', alpha=0.7)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Logo Class Distribution', fontsize=14, pad=20)
    plt.xticks(x, list(train_classes.keys()), rotation=45, ha='right')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close()
    
    # 2. Bounding Box Sizes
    plt.figure(figsize=(10, 10))
    train_boxes = np.array(train_stats['label_stats']['box_sizes'])
    plt.scatter(train_boxes[:, 0], train_boxes[:, 1], alpha=0.6, label='Train', 
              c='#2196F3', s=50, edgecolors='white', linewidth=0.5)
    plt.xlabel('Width (normalized)', fontsize=12)
    plt.ylabel('Height (normalized)', fontsize=12)
    plt.title('Bounding Box Size Distribution', fontsize=14, pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'box_sizes.png')
    plt.close()
    
    # 3. Image Size Distribution
    plt.figure(figsize=(12, 6))
    train_sizes = np.array(train_stats['image_stats']['sizes'])
    plt.scatter(train_sizes[:, 0], train_sizes[:, 1], alpha=0.6,
              c='#2196F3', s=50, edgecolors='white', linewidth=0.5)
    plt.xlabel('Width (pixels)', fontsize=12)
    plt.ylabel('Height (pixels)', fontsize=12)
    plt.title('Image Size Distribution', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'image_sizes.png')
    plt.close()
    
    # 4. Boxes per Image Distribution
    plt.figure(figsize=(10, 6))
    train_boxes_per_img = list(train_stats['label_stats']['boxes_per_image'].values())
    plt.hist(train_boxes_per_img, bins=20, alpha=0.7, label='Train',
           color='#2196F3', edgecolor='white', linewidth=1)
    plt.xlabel('Number of Boxes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Distribution of Logos per Image', fontsize=14, pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'boxes_per_image.png')
    plt.close()
    
    # Save detailed statistics
    stats = {
        'train': {
            'image_count': len(train_stats['image_stats']['sizes']),
            'class_distribution': train_stats['label_stats']['class_distribution'],
            'mean_boxes_per_image': train_stats['label_stats']['mean_boxes_per_image'],
            'max_boxes_per_image': train_stats['label_stats']['max_boxes_per_image'],
            'mean_image_size': {
                'width': train_stats['image_stats']['mean_width'],
                'height': train_stats['image_stats']['mean_height']
            }
        },
        'test': {
            'image_count': len(test_stats['image_stats']['sizes']),
            'class_distribution': test_stats['label_stats']['class_distribution'],
            'mean_boxes_per_image': test_stats['label_stats']['mean_boxes_per_image'],
            'max_boxes_per_image': test_stats['label_stats']['max_boxes_per_image'],
            'mean_image_size': {
                'width': test_stats['image_stats']['mean_width'],
                'height': test_stats['image_stats']['mean_height']
            }
        }
    }
    
    with open(output_dir / 'dataset_statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    return stats

def main():
    """Analyze the logo detection dataset and generate visualizations."""
    base_path = Path('data/logos-dataset-section-1/logos_dataset')
    
    # Analyze training set
    train_image_stats = analyze_image_sizes(base_path / 'images' / 'train')
    train_label_stats = analyze_labels(base_path / 'labels' / 'train')
    
    # Analyze test set
    test_image_stats = analyze_image_sizes(base_path / 'images' / 'test')
    test_label_stats = analyze_labels(base_path / 'labels' / 'test')
    
    # Generate plots and save statistics
    stats = plot_statistics(
        {'image_stats': train_image_stats, 'label_stats': train_label_stats},
        {'image_stats': test_image_stats, 'label_stats': test_label_stats}
    )
    
    # Print summary
    print("\nDataset Analysis Summary:")
    print(f"Training set: {stats['train']['image_count']} images")
    print(f"Test set: {stats['test']['image_count']} images")
    
    # Calculate total logos for percentage calculation
    train_total = sum(stats['train']['class_distribution'].values())
    test_total = sum(stats['test']['class_distribution'].values())
    
    print("\nClass distribution:")
    print("Class      | Train Count (%)      | Test Count (%)")
    print("-" * 50)
    for class_name in stats['train']['class_distribution'].keys():
        train_count = stats['train']['class_distribution'][class_name]
        test_count = stats['test']['class_distribution'].get(class_name, 0)
        train_percent = (train_count / train_total) * 100
        test_percent = (test_count / test_total) * 100
        print(f"{class_name:<10} | {train_count:>4} ({train_percent:>6.2f}%) | {test_count:>4} ({test_percent:>6.2f}%)")
    
    print(f"\nAverage logos per image (train): {stats['train']['mean_boxes_per_image']:.2f}")
    print(f"Maximum logos in a single image (train): {stats['train']['max_boxes_per_image']}")
    print("\nAnalysis complete! Check the 'results/analysis' directory for visualizations.")

if __name__ == "__main__":
    main()

