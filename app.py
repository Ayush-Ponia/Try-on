import streamlit as st
import os
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import cv2
import sys
import tempfile
import shutil
from io import BytesIO

# Define the Generator class directly in this file to avoid import issues
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the generator model architecture (copied from fixed_code.py)
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.relu(out)

class Generator(nn.Module):
    def __init__(self, input_channels=23, output_channels=3):
        super(Generator, self).__init__()
        
        # Initial convolution block
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, padding=3)
        self.norm1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Downsampling blocks
        self.down1 = self._downsample(64, 128)
        self.down2 = self._downsample(128, 256)
        self.down3 = self._downsample(256, 512)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(512) for _ in range(9)]
        )
        
        # Upsampling blocks
        self.up1 = self._upsample(512, 256)
        self.up2 = self._upsample(256, 128)
        self.up3 = self._upsample(128, 64)
        
        # Output convolution
        self.conv_out = nn.Conv2d(64, output_channels, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()
        
    def _downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Initial convolution
        x = self.relu(self.norm1(self.conv1(x)))
        
        # Downsampling
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        # Residual blocks
        x = self.res_blocks(x)
        
        # Upsampling
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        # Output
        x = self.tanh(self.conv_out(x))
        return x

# Set page config
st.set_page_config(
    page_title="VITON-HD Virtual Try-On",
    page_icon="👚",
    layout="wide"
)

# Application title and description
st.title("Virtual Try-On with VITON-HD")
st.markdown("""
Upload your photo and a clothing item to see how the clothing looks on you!
This application uses the VITON-HD deep learning model to generate realistic virtual try-on images.
""")

# Functions for image processing
def preprocess_person_image(image):
    # Resize to 512x384
    image = image.resize((384, 512), Image.LANCZOS)
    return image

def preprocess_cloth_image(image):
    # Resize to 512x384
    image = image.resize((384, 512), Image.LANCZOS)
    return image

def generate_cloth_mask(cloth_image):
    # Convert to numpy array
    cloth = np.array(cloth_image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cloth, cv2.COLOR_RGB2GRAY)
    
    # Threshold to get binary mask (adjust threshold as needed)
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    # Create PIL image from numpy array
    mask_image = Image.fromarray(mask)
    return mask_image

def create_pose_keypoints(height=512, width=384):
    """Create default pose keypoints when not available"""
    # Create a simple default pose (standing position)
    pose_data = {"people": [{"pose_keypoints_2d": []}]}
    
    # Default keypoints (x, y, confidence) for a basic standing pose
    # Format: [neck_x, neck_y, 1, right_shoulder_x, right_shoulder_y, 1, ...]
    default_keypoints = [
        # Neck
        width // 2, height // 4, 1,
        # Right shoulder
        width // 3, height // 4, 1,
        # Right elbow
        width // 4, height // 3, 1,
        # Right wrist
        width // 4, height // 2, 1,
        # Left shoulder
        2 * width // 3, height // 4, 1,
        # Left elbow
        3 * width // 4, height // 3, 1,
        # Left wrist
        3 * width // 4, height // 2, 1,
        # Right hip
        width // 3, 2 * height // 3, 1,
        # Right knee
        width // 3, 3 * height // 4, 1,
        # Right ankle
        width // 3, height - 20, 1,
        # Left hip
        2 * width // 3, 2 * height // 3, 1,
        # Left knee
        2 * width // 3, 3 * height // 4, 1,
        # Left ankle
        2 * width // 3, height - 20, 1,
        # Face points and others (simplified)
        width // 2, height // 8, 1,
        width // 2 - 10, height // 8, 1,
        width // 2 + 10, height // 8, 1,
        width // 2, height // 7, 1,
        width // 2, height // 6, 1
    ]
    
    pose_data["people"][0]["pose_keypoints_2d"] = default_keypoints
    return pose_data

def create_simple_parse(person_image):
    """Create a simple segmentation mask when not available"""
    # Create a basic segmentation with upper body and lower body regions
    width, height = person_image.size
    parse = np.zeros((height, width), dtype=np.uint8)
    
    # Simple segmentation:
    # Upper body (top 40%) - value 5 (upper clothes)
    parse[:int(height * 0.4), :] = 5
    
    # Lower body (bottom 60%) - value 9 (pants)
    parse[int(height * 0.4):, :] = 9
    
    # Create PIL image from numpy array
    parse_image = Image.fromarray(parse)
    return parse_image

def prepare_input_data(person_image, cloth_image):
    """Prepare all the necessary inputs for the VITON-HD model"""
    # Create a temporary directory to store the dataset structure
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create the dataset structure
        test_dir = os.path.join(temp_dir, 'test')
        os.makedirs(os.path.join(test_dir, 'image'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'cloth'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'cloth-mask'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'image-parse'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'pose'), exist_ok=True)
        
        # Process and save the images
        # Generate a random filename
        filename = 'test_image.jpg'
        
        # Save person image
        person_image_processed = preprocess_person_image(person_image)
        person_image_path = os.path.join(test_dir, 'image', filename)
        person_image_processed.save(person_image_path)
        
        # Save cloth image
        cloth_image_processed = preprocess_cloth_image(cloth_image)
        cloth_image_path = os.path.join(test_dir, 'cloth', filename)
        cloth_image_processed.save(cloth_image_path)
        
        # Generate and save cloth mask
        cloth_mask = generate_cloth_mask(cloth_image_processed)
        cloth_mask_path = os.path.join(test_dir, 'cloth-mask', filename)
        cloth_mask.save(cloth_mask_path)
        
        # Create and save simple parse image
        parse_image = create_simple_parse(person_image_processed)
        parse_path = os.path.join(test_dir, 'image-parse', filename.replace('.jpg', '.png'))
        parse_image.save(parse_path)
        
        # Create and save pose keypoints
        pose_data = create_pose_keypoints()
        pose_path = os.path.join(test_dir, 'pose', filename.replace('.jpg', '_keypoints.json'))
        with open(pose_path, 'w') as f:
            json.dump(pose_data, f)
        
        return temp_dir, filename
    
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

# Function to load model and perform virtual try-on
def perform_virtual_tryon(generator_path, person_image, cloth_image):
    """Load the VITON-HD model and generate try-on image"""
    # Initialize the generator model
    generator = Generator().to(device)
    
    # Load pre-trained weights
    try:
        # Handle different PyTorch versions
        try:
            generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
        except Exception as torch_error:
            st.warning(f"Standard loading failed, trying alternative method: {torch_error}")
            # Try alternative loading method for different PyTorch versions
            checkpoint = torch.load(generator_path, map_location=lambda storage, loc: storage)
            generator.load_state_dict(checkpoint)
        
        generator.eval()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    
    # Prepare input data
    try:
        temp_dir, filename = prepare_input_data(person_image, cloth_image)
        
        # Set up transformations
        rgb_transform = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor()
        ])
        
        # Load processed images
        person_path = os.path.join(temp_dir, 'test', 'image', filename)
        cloth_path = os.path.join(temp_dir, 'test', 'cloth', filename)
        cloth_mask_path = os.path.join(temp_dir, 'test', 'cloth-mask', filename)
        parse_path = os.path.join(temp_dir, 'test', 'image-parse', filename.replace('.jpg', '.png'))
        pose_path = os.path.join(temp_dir, 'test', 'pose', filename.replace('.jpg', '_keypoints.json'))
        
        # Load images
        person_img = Image.open(person_path).convert('RGB')
        cloth_img = Image.open(cloth_path).convert('RGB')
        cloth_mask_img = Image.open(cloth_mask_path).convert('L')
        parse_img = Image.open(parse_path).convert('L')
        
        # Apply transformations
        person_tensor = rgb_transform(person_img).unsqueeze(0).to(device)
        cloth_tensor = rgb_transform(cloth_img).unsqueeze(0).to(device)
        cloth_mask_tensor = mask_transform(cloth_mask_img).unsqueeze(0).to(device)
        parse_tensor = mask_transform(parse_img).unsqueeze(0).to(device)
        
        # Load pose data and convert to tensor
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        
        # Create pose map
        pose_map = np.zeros((18, 512, 384), dtype=np.float32)
        
        if 'people' in pose_data and len(pose_data['people']) > 0:
            keypoints = pose_data['people'][0]['pose_keypoints_2d']
            for i in range(18):  # 18 keypoints in COCO format
                x, y, confidence = keypoints[i*3], keypoints[i*3+1], keypoints[i*3+2]
                if confidence > 0:
                    x = int(x)
                    y = int(y)
                    
                    if 0 <= x < 384 and 0 <= y < 512:
                        # Create a simple heatmap
                        sigma = 6
                        for dx in range(-sigma, sigma + 1):
                            for dy in range(-sigma, sigma + 1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < 384 and 0 <= ny < 512:
                                    d = (dx * dx + dy * dy) / (2 * sigma * sigma)
                                    pose_map[i, ny, nx] = np.exp(-d)
        
        pose_tensor = torch.from_numpy(pose_map).float().unsqueeze(0).to(device)
        
        # Concatenate all inputs
        input_tensor = torch.cat([cloth_tensor, cloth_mask_tensor, parse_tensor, pose_tensor], dim=1)
        
        # Generate try-on image
        with torch.no_grad():
            output = generator(input_tensor)
        
        # Convert output tensor to image
        output_np = output.squeeze().cpu().numpy()
        output_np = np.transpose(output_np, (1, 2, 0))
        output_np = (output_np * 0.5 + 0.5) * 255
        output_np = output_np.astype(np.uint8)
        output_img = Image.fromarray(output_np)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        return output_img
        
    except Exception as e:
        st.error(f"Error during virtual try-on: {e}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return None

# Sidebar for model selection and upload
with st.sidebar:
    st.header("Model Settings")
    
    # Option to upload a pre-trained model
    st.subheader("Upload Pre-trained Model")
    uploaded_model = st.file_uploader("Upload model weights (.pth file)", type=["pth"])
    
    # Or use a demo model if available
    st.subheader("Or use existing model")
    model_path = st.text_input("Path to pre-trained model (.pth file)", "checkpoints/generator_epoch_205.pth")
    
    # Model selection help text
    st.info("If you don't have a pre-trained model, you'll need to train one using the VITON-HD training script first, or you can upload a pre-trained model file (.pth).")
    
    # Save uploaded model to temp file if provided
    if uploaded_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            model_path = tmp_file.name
            st.success("Model uploaded successfully!")

# Main content area - split into two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Images")
    
    # Person image upload
    st.subheader("Your Photo")
    person_image = st.file_uploader("Upload a full-body photo of yourself", type=["jpg", "jpeg", "png"])
    
    if person_image is not None:
        person_img = Image.open(person_image).convert('RGB')
        st.image(person_img, caption="Your Photo", use_column_width=True)
    
    # Cloth image upload
    st.subheader("Clothing Item")
    cloth_image = st.file_uploader("Upload a clothing item (front view on plain background)", type=["jpg", "jpeg", "png"])
    
    if cloth_image is not None:
        cloth_img = Image.open(cloth_image).convert('RGB')
        st.image(cloth_img, caption="Clothing Item", use_column_width=True)

# Generate button
if person_image is not None and cloth_image is not None:
    if st.button("Generate Virtual Try-On", type="primary"):
        with st.spinner("Generating virtual try-on image..."):
            try:
                # Check if model path exists for local files
                if (not uploaded_model and not os.path.exists(model_path)):
                    st.error(f"Model file not found at {model_path}. Please make sure the model exists or upload a model file.")
                else:
                    # Open image files
                    person_img = Image.open(person_image).convert('RGB')
                    cloth_img = Image.open(cloth_image).convert('RGB')
                    
                    # Add a progress bar
                    progress_bar = st.progress(0)
                    st.markdown("Preparing images...")
                    progress_bar.progress(25)
                    
                    # Update progress
                    st.markdown("Loading model...")
                    progress_bar.progress(50)
                    
                    # Perform virtual try-on
                    result_img = perform_virtual_tryon(model_path, person_img, cloth_img)
                    progress_bar.progress(100)
                    
                    if result_img is not None:
                        # Display the result
                        with col2:
                            st.header("Virtual Try-On Result")
                            st.image(result_img, caption="Try-On Result", use_column_width=True)
                            
                            # Add download button for the result
                            buffered = BytesIO()
                            result_img.save(buffered, format="PNG")
                            st.download_button(
                                label="Download Result",
                                data=buffered.getvalue(),
                                file_name="virtual_tryon_result.png",
                                mime="image/png"
                            )
                    else:
                        st.error("Failed to generate try-on image. Please try again with different images.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("If you're seeing PyTorch-related errors, please make sure you have the correct version installed.")
else:
    with col2:
        st.header("Virtual Try-On Result")
        st.info("Please upload both a person photo and a clothing item to generate a virtual try-on image.")

# Instructions and tips
st.markdown("""
## Tips for Best Results
1. **Person Photo**:
   - Use a full-body photo with a clean background
   - Stand in a neutral pose facing the camera
   - Wear tight-fitting clothes for better body shape detection

2. **Clothing Item**:
   - Use front-view images of clothing on a plain background
   - T-shirts, shirts, and jackets work best
   - The clothing should be clearly visible and well-lit

3. **Model Weights**:
   - For best results, use a pre-trained model specifically trained on similar clothing types
   - If you don't have a pre-trained model, you can train one using the VITON-HD training script
""")

# Add info about the model
st.markdown("""
## About VITON-HD
VITON-HD is a virtual try-on model that uses deep learning to transfer clothing items onto a person's image. 
The model works by analyzing both the person's body pose and the clothing item, then generating a realistic 
composite that shows how the clothing would look when worn by the person.

This app implements the VITON-HD architecture with a user-friendly interface for easy virtual try-on.
""")

# Footer
st.markdown("---")
st.markdown("Virtual Try-On Demo using VITON-HD | Built with Streamlit")
