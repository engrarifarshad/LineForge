import cv2
import numpy as np
import random
import math

def draw_lines_from_image(img, num_lines=5000, line_thickness=1, edge_threshold1=50, edge_threshold2=150):
    """
    Create an image using straight lines that preserve high and low level features.
    Uses intensity-based density sampling and adaptive hatching.
    
    Parameters:
    - img: Input grayscale image
    - num_lines: Number of lines to draw
    - line_thickness: Thickness of lines
    - edge_threshold1: Lower threshold for Canny edge detection
    - edge_threshold2: Upper threshold for Canny edge detection
    """
    # Get image dimensions
    height, width = img.shape
    
    # Normalize image to 0-255 range
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Create new image array with all pixels set to 0 (black background)
    new_img = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate gradient magnitude for feature detection
    grad_x = cv2.Sobel(img_normalized, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_normalized, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Detect edges
    edges = cv2.Canny(img_normalized, edge_threshold1, edge_threshold2)
    edge_points = np.column_stack(np.where(edges > 0))
    
    # Create importance map: combine intensity contrast and gradient
    # Invert intensity so darker areas get higher importance (more lines)
    inverted_intensity = 255 - img_normalized
    importance_map = (inverted_intensity.astype(float) * 0.6 + gradient_magnitude.astype(float) * 0.4).astype(np.uint8)
    
    # Normalize importance map for probability sampling
    importance_flat = importance_map.flatten().astype(float)
    importance_flat = importance_flat / (importance_flat.sum() + 1e-10)
    
    # Generate all pixel coordinates
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    coords = np.column_stack([y_coords.ravel(), x_coords.ravel()])
    
    # Method 1: Intensity-based hatching lines (preserve tone)
    # Use multiple orientations for better tone representation
    num_hatching_lines = int(num_lines * 0.6)
    orientations = [0, 45, 90, 135]  # Different line angles
    
    for orientation in orientations:
        num_lines_this_orientation = num_hatching_lines // len(orientations)
        angle_rad = math.radians(orientation)
        
        for i in range(num_lines_this_orientation):
            # Sample point based on importance (more lines in important areas)
            idx = np.random.choice(len(coords), p=importance_flat)
            y, x = coords[idx]
            
            # Get local intensity to determine line properties
            local_intensity = img_normalized[y, x]
            
            # Line length varies with local contrast
            local_gradient = gradient_magnitude[y, x]
            base_length = min(width, height) // 8
            line_length = int(base_length * (1 + local_gradient / 255.0 * 0.5))
            line_length = max(5, min(line_length, min(width, height) // 3))
            
            # Calculate line endpoints
            half_len = line_length / 2
            start_x = int(x - half_len * math.cos(angle_rad))
            start_y = int(y - half_len * math.sin(angle_rad))
            end_x = int(x + half_len * math.cos(angle_rad))
            end_y = int(y + half_len * math.sin(angle_rad))
            
            # Clamp to image bounds
            start_x = max(0, min(width - 1, start_x))
            start_y = max(0, min(height - 1, start_y))
            end_x = max(0, min(width - 1, end_x))
            end_y = max(0, min(height - 1, end_y))
            
            # Line color based on local intensity
            line_color = int(local_intensity)
            
            cv2.line(new_img, (start_x, start_y), (end_x, end_y), line_color, line_thickness)
    
    # Method 2: Edge-following lines (preserve structure)
    num_edge_lines = int(num_lines * 0.25)
    
    if len(edge_points) > 0:
        # Limit edge points for performance
        if len(edge_points) > 5000:
            indices = np.random.choice(len(edge_points), 5000, replace=False)
            edge_points = edge_points[indices]
        
        for i in range(num_edge_lines):
            idx = random.randint(0, len(edge_points) - 1)
            y, x = edge_points[idx]
            
            # Get gradient direction
            if 0 < x < width - 1 and 0 < y < height - 1:
                gx = grad_x[y, x]
                gy = grad_y[y, x]
                
                if abs(gx) > 1e-6 or abs(gy) > 1e-6:
                    # Angle perpendicular to gradient (along edge)
                    angle = math.atan2(gy, gx) + math.pi / 2
                else:
                    angle = random.uniform(0, 2 * math.pi)
            else:
                angle = random.uniform(0, 2 * math.pi)
            
            # Line length based on local gradient strength
            local_grad = gradient_magnitude[y, x]
            line_length = int(10 + (min(width, height) // 6) * (local_grad / 255.0))
            line_length = max(5, min(line_length, min(width, height) // 4))
            
            half_len = line_length / 2
            start_x = int(x - half_len * math.cos(angle))
            start_y = int(y - half_len * math.sin(angle))
            end_x = int(x + half_len * math.cos(angle))
            end_y = int(y + half_len * math.sin(angle))
            
            # Clamp coordinates
            start_x = max(0, min(width - 1, start_x))
            start_y = max(0, min(height - 1, start_y))
            end_x = max(0, min(width - 1, end_x))
            end_y = max(0, min(height - 1, end_y))
            
            line_color = int(img_normalized[y, x])
            cv2.line(new_img, (start_x, start_y), (end_x, end_y), line_color, line_thickness)
    
    # Method 3: Connect nearby high-contrast points (preserve features)
    num_connection_lines = num_lines - num_hatching_lines - num_edge_lines
    
    # Sample points from high-contrast regions
    high_contrast_threshold = np.percentile(importance_map, 75)
    high_contrast_points = coords[importance_map.flatten() > high_contrast_threshold]
    
    if len(high_contrast_points) > 10000:
        indices = np.random.choice(len(high_contrast_points), 10000, replace=False)
        high_contrast_points = high_contrast_points[indices]
    
    for i in range(num_connection_lines):
        if len(high_contrast_points) < 2:
            break
        
        # Select two points
        idx1, idx2 = random.sample(range(len(high_contrast_points)), 2)
        y1, x1 = high_contrast_points[idx1]
        y2, x2 = high_contrast_points[idx2]
        
        # Calculate distance
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        max_dist = min(width, height) * 0.25
        
        if dist < max_dist and dist > 5:
            # Sample intensities along the line
            num_samples = max(10, int(dist))
            x_coords = np.linspace(x1, x2, num_samples)
            y_coords = np.linspace(y1, y2, num_samples)
            
            intensities = []
            for x, y in zip(x_coords, y_coords):
                x, y = int(x), int(y)
                if 0 <= x < width and 0 <= y < height:
                    intensities.append(img_normalized[y, x])
            
            if intensities:
                # Use average intensity, but preserve contrast
                avg_intensity = np.mean(intensities)
                line_color = int(avg_intensity)
                
                cv2.line(new_img, (int(x1), int(y1)), (int(x2), int(y2)), line_color, line_thickness)
    
    return new_img
image_path = '1.png'
# Read input image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image '{image_path}'. Please check if the file exists.")
else:
    # Create line art image
    print("Creating line art image...")
    new_img = draw_lines_from_image(img, num_lines=50000, line_thickness=1)
    
    # Save the result
    cv2.imwrite('new_image.png', new_img)
    print("Line art image saved as 'new_image.png'")
