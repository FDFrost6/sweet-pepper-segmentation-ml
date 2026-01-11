import pickle
import matplotlib.pyplot as plt
import numpy as np 

def plot_q1_color_data():
    """Plot Q1 color data as scatter plots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, fname in enumerate(['Q1_BG_dict.pkl', 'Q1_Red_dict.pkl', 'Q1_Yellow_dict.pkl']):
        print(f"Plotting color data from: {fname}")
        with open(fname, 'rb') as fp:
            data = pickle.load(fp)
            
        # Use training data for visualization
        train_data = data['train']
        
        # Plot 3D scatter if 3 features, otherwise 2D
        if train_data.shape[1] == 3:
            # Sample some points for better visualization
            sample_size = min(1000, train_data.shape[0])
            sample_indices = np.random.choice(train_data.shape[0], sample_size, replace=False)
            sample_data = train_data[sample_indices]
            
            # Create 2D projection (R vs G)
            axes[i].scatter(sample_data[:, 0], sample_data[:, 1], 
                          c=sample_data/255.0, s=1, alpha=0.6)
            axes[i].set_xlabel('Feature 1 (R)')
            axes[i].set_ylabel('Feature 2 (G)')
            axes[i].set_title(f'{fname.split("_")[1]} Color Data')
            
    plt.tight_layout()
    plt.show()

def plot_q2_images():
    """Plot sample images from Q2 data"""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for i, fname in enumerate(['Q2_BG_dict.pkl', 'Q2_SP_dict.pkl']):
        print(f"Plotting images from: {fname}")
        with open(fname, 'rb') as fp:
            data = pickle.load(fp)
            
        # Get training data
        train_images = data['train']
        
        # Plot first 6 images from training set
        for j in range(6):
            if j < len(train_images):
                # Normalize pixel values to [0,1] if they're in [0,255]
                img = train_images[j]
                if img.max() > 1.0:
                    img = img / 255.0
                    
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                axes[i, j].set_title(f'{fname.split("_")[1]} - Image {j+1}')
            else:
                axes[i, j].axis('off')
                
    plt.tight_layout()
    plt.show()

def plot_image_statistics():
    """Plot statistics about the images"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for i, fname in enumerate(['Q2_BG_dict.pkl', 'Q2_SP_dict.pkl']):
        with open(fname, 'rb') as fp:
            data = pickle.load(fp)
            
        train_images = np.array(data['train'])
        
        # Calculate mean image
        mean_img = np.mean(train_images, axis=0)
        if mean_img.max() > 1.0:
            mean_img = mean_img / 255.0
            
        # Calculate standard deviation
        std_img = np.std(train_images, axis=0)
        if train_images.max() > 1.0:
            std_img = std_img / 255.0
            
        # Plot mean image
        axes[i, 0].imshow(mean_img)
        axes[i, 0].set_title(f'{fname.split("_")[1]} - Mean Image')
        axes[i, 0].axis('off')
        
        # Plot std image
        axes[i, 1].imshow(std_img, cmap='gray')
        axes[i, 1].set_title(f'{fname.split("_")[1]} - Std Dev')
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_q1_as_images():
    """Plot Q1 color data as images by arranging RGB values in a grid"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, fname in enumerate(['Q1_BG_dict.pkl', 'Q1_Red_dict.pkl', 'Q1_Yellow_dict.pkl']):
        print(f"Creating image from color data: {fname}")
        with open(fname, 'rb') as fp:
            data = pickle.load(fp)
            
        # Use training data
        train_data = data['train']
        
        # Sample data for visualization (create a square grid)
        n_points = min(10000, train_data.shape[0])  # Use up to 10k points
        grid_size = int(np.sqrt(n_points))  # Make it square
        n_points = grid_size * grid_size
        
        # Sample random points
        sample_indices = np.random.choice(train_data.shape[0], n_points, replace=False)
        sample_data = train_data[sample_indices]
        
        # Reshape into grid and normalize to [0,1]
        img_data = sample_data.reshape(grid_size, grid_size, 3)
        if img_data.max() > 1.0:
            img_data = img_data / 255.0
            
        # Plot as image
        axes[i].imshow(img_data)
        axes[i].set_title(f'{fname.split("_")[1]} Color Data as Image ({grid_size}x{grid_size})')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_q1_color_distribution():
    """Plot Q1 color data as histograms showing RGB distribution"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    color_names = ['BG', 'Red', 'Yellow']
    rgb_labels = ['Red Channel', 'Green Channel', 'Blue Channel']
    colors = ['red', 'green', 'blue']
    
    for i, fname in enumerate(['Q1_BG_dict.pkl', 'Q1_Red_dict.pkl', 'Q1_Yellow_dict.pkl']):
        with open(fname, 'rb') as fp:
            data = pickle.load(fp)
            
        train_data = data['train']
        
        # Plot histogram for each RGB channel
        for j in range(3):
            axes[i, j].hist(train_data[:, j], bins=50, color=colors[j], alpha=0.7, density=True)
            axes[i, j].set_title(f'{color_names[i]} - {rgb_labels[j]}')
            axes[i, j].set_xlabel('Pixel Value')
            axes[i, j].set_ylabel('Density')
            axes[i, j].grid(True, alpha=0.3)
            
    plt.tight_layout()
    plt.show()

# Print data information
print("=== Data Information ===")
for fname in ['Q1_BG_dict.pkl', 'Q1_Red_dict.pkl', 'Q1_Yellow_dict.pkl']:
	print(f"For the file: {fname}")
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print(f"  The sets in the dictionary are: {data.keys()}")
		print(f"  The size of the data matrix X for each set is: {data['train'].shape}, {data['validation'].shape}, {data['evaluation'].shape}")

for fname in ['Q2_BG_dict.pkl', 'Q2_SP_dict.pkl']:
	print(f"For the file: {fname}")
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print(f"  The sets in the dictionary are: {data.keys()}")
		print(f"  Number of images for each set: {len(data['train'])}, {len(data['validation'])}, {len(data['evaluation'])}")
		print(f"  Image dimensions: {data['train'][0].shape}")

print("\n=== Plotting Data ===")

# Plot Q1 color data
print("Plotting Q1 color data...")
plot_q1_color_data()

# Plot Q2 image data
print("Plotting Q2 sample images...")
plot_q2_images()

print("Plotting Q2 image statistics...")
plot_image_statistics()

print("Plotting Q1 color data as images...")
plot_q1_as_images()

print("Plotting Q1 color distribution...")
plot_q1_color_distribution()

print("Plotting Q1 color data as images...")
plot_q1_as_images()

print("Plotting Q1 color distribution...")
plot_q1_color_distribution()