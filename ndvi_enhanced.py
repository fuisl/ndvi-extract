#!/usr/bin/env python3
"""
Enhanced NDVI Imaging System
Complete implementation based on research by Hoang (2025), Singh et al. (2020), and Stamford et al. (2023)

Features:
- Dual camera synchronized capture (RGB + NoIR with 720nm filter)
- 6-reference calibration system for accurate reflectance
- Automated stereo rectification and alignment
- Proper NIR extraction using R-B difference method
- IoT connectivity preparation
- Comprehensive validation and quality checks
"""

import cv2
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
import yaml

# Camera imports
try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Warning: Picamera2 not available. Using mock camera for testing.")

# Configuration
@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    resolution: Tuple[int, int] = (1920, 1080)
    format: str = "RGB888"
    iso: int = 400
    shutter_speed_rgb: int = 400  # microseconds
    shutter_speed_nir: int = 2500  # microseconds
    awb_enable: bool = False
    ae_enable: bool = True

@dataclass
class CalibrationTarget:
    """Calibration reference target specification"""
    name: str
    red_reflectance: float  # Percentage (0-100)
    nir_reflectance: float  # Percentage (0-100)
    position: Optional[Tuple[int, int]] = None  # Pixel position when detected

@dataclass
class NDVIConfig:
    """NDVI system configuration"""
    red_wavelength: int = 620  # nm
    nir_wavelength: int = 750  # nm
    filter_cutoff: int = 720   # nm (long-pass filter)
    nir_leakage_correction: float = 0.05  # Correction for red light leakage through filter

class MockCamera:
    """Mock camera for testing when Picamera2 is not available"""
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        
    def start(self):
        pass
        
    def stop(self):
        pass
        
    def capture_array(self):
        # Return a test pattern
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    def set_controls(self, controls: dict):
        pass
    
    def configure(self, config):
        pass

class NDVISystem:
    """Complete NDVI imaging system implementation"""
    
    def __init__(self, config_file: str = None):
        """Initialize NDVI system with configuration"""
        self.setup_logging()
        self.load_config(config_file)
        self.setup_directories()
        self.cameras = {}
        self.calibration_data = None
        self.stereo_params = None
        
        # Initialize calibration targets (based on Stamford et al. 6-reference method)
        self.calibration_targets = [
            CalibrationTarget("White", 92.69, 87.62),
            CalibrationTarget("Sand", 63.59, 60.85),
            CalibrationTarget("Brown", 16.29, 16.13),
            CalibrationTarget("Indian_Birch", 66.72, 62.69),
            CalibrationTarget("Forest_Green", 4.90, 5.88),
            CalibrationTarget("Burgundy", 25.53, 39.40)
        ]
        
        self.logger.info("NDVI System initialized")
    
    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ndvi_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: str = None):
        """Load system configuration"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.camera_config = CameraConfig(**config_dict.get('camera', {}))
            self.ndvi_config = NDVIConfig(**config_dict.get('ndvi', {}))
        else:
            self.camera_config = CameraConfig()
            self.ndvi_config = NDVIConfig()
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data/raw/rgb',
            'data/raw/nir',
            'data/processed/aligned',
            'data/processed/ndvi',
            'data/calibration',
            'data/validation',
            'logs',
            'config'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def initialize_cameras(self) -> bool:
        """Initialize dual camera system"""
        try:
            if PICAMERA_AVAILABLE:
                # Camera 0: RGB camera
                self.cameras['rgb'] = Picamera2(0)
                rgb_config = self.cameras['rgb'].create_video_configuration(
                    main={"size": self.camera_config.resolution, "format": self.camera_config.format}
                )
                self.cameras['rgb'].configure(rgb_config)
                
                # RGB camera settings
                self.cameras['rgb'].set_controls({
                    "AwbEnable": self.camera_config.awb_enable,
                    "AeEnable": self.camera_config.ae_enable,
                    "ExposureTime": self.camera_config.shutter_speed_rgb,
                    "AnalogueGain": 1.0
                })
                
                # Camera 1: NoIR camera with 720nm filter
                self.cameras['nir'] = Picamera2(1)
                nir_config = self.cameras['nir'].create_video_configuration(
                    main={"size": self.camera_config.resolution, "format": self.camera_config.format}
                )
                self.cameras['nir'].configure(nir_config)
                
                # NIR camera settings (higher exposure due to filter)
                self.cameras['nir'].set_controls({
                    "AwbEnable": False,  # Disable white balance for NIR
                    "AeEnable": False,   # Manual exposure for consistency
                    "ExposureTime": self.camera_config.shutter_speed_nir,
                    "AnalogueGain": 2.0  # Higher gain for NIR sensitivity
                })
                
                # Start cameras
                self.cameras['rgb'].start()
                self.cameras['nir'].start()
                
            else:
                # Use mock cameras for testing
                self.cameras['rgb'] = MockCamera(0)
                self.cameras['nir'] = MockCamera(1)
                self.cameras['rgb'].start()
                self.cameras['nir'].start()
            
            self.logger.info("Cameras initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cameras: {e}")
            return False
    
    def capture_synchronized_images(self, save_raw: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Capture synchronized images from both cameras"""
        try:
            # Capture from both cameras simultaneously
            rgb_image = self.cameras['rgb'].capture_array()
            nir_image = self.cameras['nir'].capture_array()
            
            if save_raw:
                timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
                
                # Save raw images
                cv2.imwrite(f"data/raw/rgb/{timestamp}_RGB.jpg", 
                           cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"data/raw/nir/{timestamp}_NIR.jpg", 
                           cv2.cvtColor(nir_image, cv2.COLOR_RGB2BGR))
                
                self.logger.info(f"Captured synchronized images: {timestamp}")
            
            return rgb_image, nir_image
            
        except Exception as e:
            self.logger.error(f"Failed to capture images: {e}")
            return None, None
    
    def extract_nir_signal(self, nir_image: np.ndarray) -> np.ndarray:
        """
        Extract NIR signal using R-B difference method (Hoang's methodology)
        Accounts for 720nm long-pass filter characteristics
        """
        # Convert to float for processing
        nir_float = nir_image.astype(np.float32)
        
        # Extract color channels
        red_channel = nir_float[:, :, 0]  # Red channel (contains NIR + some red leakage)
        blue_channel = nir_float[:, :, 2]  # Blue channel (minimal NIR, some red leakage)
        
        # R-B difference removes common leakage and enhances NIR signal
        nir_signal = red_channel - blue_channel
        
        # Apply leakage correction based on filter specifications
        nir_signal *= (1 - self.ndvi_config.nir_leakage_correction)
        
        # Ensure positive values
        nir_signal = np.maximum(nir_signal, 0)
        
        return nir_signal
    
    def detect_calibration_targets(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect calibration targets in image using template matching or color detection
        This is a simplified implementation - in practice, you'd use more sophisticated detection
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use HoughCircles to detect circular calibration targets
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        
        target_positions = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Sort by position to match with known targets
            circles = sorted(circles, key=lambda x: (x[1], x[0]))  # Sort by y, then x
            
            for (x, y, r) in circles[:len(self.calibration_targets)]:
                target_positions.append((x, y))
        
        return target_positions
    
    def calibrate_reflectance(self, image: np.ndarray, channel_type: str) -> np.ndarray:
        """
        Calibrate image to reflectance using 6-reference method (Stamford et al.)
        """
        if self.calibration_data is None:
            self.logger.warning("No calibration data available. Using default calibration.")
            return image.astype(np.float32) / 255.0 * 100  # Simple normalization
        
        # Get calibration curve for this channel
        calibration_curve = self.calibration_data.get(channel_type)
        if calibration_curve is None:
            self.logger.warning(f"No calibration curve for {channel_type}")
            return image.astype(np.float32) / 255.0 * 100
        
        # Apply calibration curve to convert digital numbers to reflectance
        image_float = image.astype(np.float32)
        
        # Handle potential gamma correction (Stamford et al. methodology)
        # Apply inverse gamma correction first
        gamma_corrected = np.power(image_float / 255.0, 2.2) * 255.0
        
        # Apply calibration interpolation
        calibrated = np.zeros_like(gamma_corrected)
        for i in range(image.shape[2] if len(image.shape) == 3 else 1):
            if len(image.shape) == 3:
                channel_data = gamma_corrected[:, :, i].flatten()
            else:
                channel_data = gamma_corrected.flatten()
            
            # Interpolate using calibration curve
            calibrated_channel = np.interp(channel_data, 
                                         calibration_curve['digital_numbers'],
                                         calibration_curve['reflectances'])
            
            if len(image.shape) == 3:
                calibrated[:, :, i] = calibrated_channel.reshape(image.shape[:2])
            else:
                calibrated = calibrated_channel.reshape(image.shape)
        
        return calibrated
    
    def perform_calibration(self, rgb_image: np.ndarray, nir_image: np.ndarray) -> bool:
        """
        Perform system calibration using reference targets
        """
        try:
            # Detect calibration targets in both images
            rgb_targets = self.detect_calibration_targets(rgb_image)
            nir_targets = self.detect_calibration_targets(nir_image)
            
            if len(rgb_targets) < 4 or len(nir_targets) < 4:
                self.logger.error("Insufficient calibration targets detected")
                return False
            
            # Extract digital numbers from target positions
            calibration_data = {
                'red': {'digital_numbers': [], 'reflectances': []},
                'nir': {'digital_numbers': [], 'reflectances': []}
            }
            
            # Process RGB targets (red channel)
            for i, (x, y) in enumerate(rgb_targets[:len(self.calibration_targets)]):
                if i < len(self.calibration_targets):
                    # Extract average digital number from target area
                    target_region = rgb_image[y-5:y+5, x-5:x+5, 0]  # Red channel
                    digital_number = np.mean(target_region)
                    
                    calibration_data['red']['digital_numbers'].append(digital_number)
                    calibration_data['red']['reflectances'].append(
                        self.calibration_targets[i].red_reflectance
                    )
            
            # Process NIR targets
            nir_signal = self.extract_nir_signal(nir_image)
            for i, (x, y) in enumerate(nir_targets[:len(self.calibration_targets)]):
                if i < len(self.calibration_targets):
                    # Extract average digital number from target area
                    target_region = nir_signal[y-5:y+5, x-5:x+5]
                    digital_number = np.mean(target_region)
                    
                    calibration_data['nir']['digital_numbers'].append(digital_number)
                    calibration_data['nir']['reflectances'].append(
                        self.calibration_targets[i].nir_reflectance
                    )
            
            # Store calibration data
            self.calibration_data = calibration_data
            
            # Save calibration data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"data/calibration/calibration_{timestamp}.json", 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            self.logger.info("Calibration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False
    
    def compute_stereo_rectification(self, rgb_image: np.ndarray, nir_image: np.ndarray) -> bool:
        """
        Compute stereo rectification parameters using automatic feature matching
        """
        try:
            # Convert to grayscale
            rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            nir_gray = cv2.cvtColor(nir_image, cv2.COLOR_RGB2GRAY)
            
            # Use ORB detector for feature matching (robust for different spectral ranges)
            orb = cv2.ORB_create(nfeatures=1000)
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(rgb_gray, None)
            kp2, des2 = orb.detectAndCompute(nir_gray, None)
            
            if des1 is None or des2 is None:
                self.logger.error("No features detected for stereo rectification")
                return False
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < 20:
                self.logger.error("Insufficient feature matches for rectification")
                return False
            
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Compute homography using RANSAC
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, 
                cv2.RANSAC, 
                ransacReprojThreshold=5.0
            )
            
            if homography is None:
                self.logger.error("Failed to compute homography")
                return False
            
            # Store stereo parameters
            self.stereo_params = {
                'homography': homography,
                'image_size': rgb_image.shape[:2]
            }
            
            # Save stereo parameters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            np.save(f"data/calibration/stereo_params_{timestamp}.npy", self.stereo_params)
            
            self.logger.info("Stereo rectification parameters computed")
            return True
            
        except Exception as e:
            self.logger.error(f"Stereo rectification failed: {e}")
            return False
    
    def align_images(self, rgb_image: np.ndarray, nir_image: np.ndarray) -> np.ndarray:
        """
        Align RGB image to NIR image using computed homography
        """
        if self.stereo_params is None:
            self.logger.warning("No stereo parameters available. Computing on-the-fly.")
            if not self.compute_stereo_rectification(rgb_image, nir_image):
                self.logger.error("Failed to compute alignment parameters")
                return rgb_image  # Return unaligned image
        
        try:
            # Apply homography transformation
            aligned_rgb = cv2.warpPerspective(
                rgb_image,
                self.stereo_params['homography'],
                (nir_image.shape[1], nir_image.shape[0])
            )
            
            return aligned_rgb
            
        except Exception as e:
            self.logger.error(f"Image alignment failed: {e}")
            return rgb_image
    
    def calculate_ndvi(self, aligned_rgb: np.ndarray, nir_image: np.ndarray, 
                      apply_calibration: bool = True) -> np.ndarray:
        """
        Calculate NDVI using proper spectral processing
        """
        try:
            # Extract red channel from aligned RGB image
            red_channel = aligned_rgb[:, :, 0].astype(np.float32)
            
            # Extract NIR signal using R-B difference method
            nir_signal = self.extract_nir_signal(nir_image)
            
            # Apply calibration if available
            if apply_calibration and self.calibration_data is not None:
                red_reflectance = self.calibrate_reflectance(red_channel, 'red')
                nir_reflectance = self.calibrate_reflectance(nir_signal, 'nir')
            else:
                # Simple normalization
                red_reflectance = red_channel / 255.0 * 100
                nir_reflectance = nir_signal / 255.0 * 100
            
            # Calculate NDVI: (NIR - Red) / (NIR + Red)
            numerator = nir_reflectance - red_reflectance
            denominator = nir_reflectance + red_reflectance + 1e-6  # Avoid division by zero
            
            ndvi = numerator / denominator
            
            # Clip NDVI values to valid range [-1, 1]
            ndvi = np.clip(ndvi, -1, 1)
            
            return ndvi
            
        except Exception as e:
            self.logger.error(f"NDVI calculation failed: {e}")
            return np.zeros_like(nir_image[:, :, 0])
    
    def create_ndvi_visualization(self, ndvi: np.ndarray, colormap: str = 'RdYlGn') -> np.ndarray:
        """
        Create color-mapped NDVI visualization
        """
        # Normalize NDVI from [-1, 1] to [0, 255]
        ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
        
        # Apply colormap
        if colormap == 'RdYlGn':
            # Custom Red-Yellow-Green colormap for vegetation
            colormap_cv = cv2.COLORMAP_RdYlGn
        else:
            colormap_cv = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_JET)
        
        ndvi_colored = cv2.applyColorMap(ndvi_normalized, colormap_cv)
        
        return ndvi_colored
    
    def process_image_pair(self, rgb_image: np.ndarray, nir_image: np.ndarray, 
                          timestamp: str = None) -> Dict:
        """
        Complete processing pipeline for an image pair
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'success': False,
            'ndvi_mean': 0.0,
            'ndvi_std': 0.0,
            'file_paths': {}
        }
        
        try:
            # Step 1: Align images
            aligned_rgb = self.align_images(rgb_image, nir_image)
            
            # Step 2: Calculate NDVI
            ndvi = self.calculate_ndvi(aligned_rgb, nir_image)
            
            # Step 3: Create visualization
            ndvi_colored = self.create_ndvi_visualization(ndvi)
            
            # Step 4: Save processed results
            base_path = f"data/processed"
            
            # Save aligned RGB
            aligned_path = f"{base_path}/aligned/{timestamp}_aligned_RGB.jpg"
            cv2.imwrite(aligned_path, cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))
            
            # Save NDVI data (as 16-bit TIFF for precision)
            ndvi_data_path = f"{base_path}/ndvi/{timestamp}_NDVI_data.tiff"
            ndvi_16bit = ((ndvi + 1) / 2 * 65535).astype(np.uint16)
            cv2.imwrite(ndvi_data_path, ndvi_16bit)
            
            # Save NDVI visualization
            ndvi_vis_path = f"{base_path}/ndvi/{timestamp}_NDVI_colored.jpg"
            cv2.imwrite(ndvi_vis_path, ndvi_colored)
            
            # Calculate statistics
            valid_mask = ~np.isnan(ndvi)
            results['ndvi_mean'] = float(np.mean(ndvi[valid_mask]))
            results['ndvi_std'] = float(np.std(ndvi[valid_mask]))
            results['valid_pixels'] = int(np.sum(valid_mask))
            results['total_pixels'] = int(ndvi.size)
            
            results['file_paths'] = {
                'aligned_rgb': aligned_path,
                'ndvi_data': ndvi_data_path,
                'ndvi_visualization': ndvi_vis_path
            }
            
            results['success'] = True
            self.logger.info(f"Image pair processed successfully: {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Image pair processing failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_capture_session(self, num_captures: int = 1, interval: float = 5.0) -> List[Dict]:
        """
        Run a complete capture and processing session
        """
        if not self.initialize_cameras():
            self.logger.error("Failed to initialize cameras")
            return []
        
        results = []
        
        try:
            for i in range(num_captures):
                self.logger.info(f"Capture {i+1}/{num_captures}")
                
                # Capture images
                rgb_image, nir_image = self.capture_synchronized_images()
                
                if rgb_image is None or nir_image is None:
                    self.logger.error(f"Failed to capture images for session {i+1}")
                    continue
                
                # Process images
                timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
                result = self.process_image_pair(rgb_image, nir_image, timestamp)
                results.append(result)
                
                # Wait for next capture
                if i < num_captures - 1:
                    import time
                    time.sleep(interval)
            
        finally:
            # Cleanup cameras
            self.cleanup_cameras()
        
        return results
    
    def cleanup_cameras(self):
        """Clean up camera resources"""
        try:
            for camera_name, camera in self.cameras.items():
                camera.stop()
                self.logger.info(f"Camera {camera_name} stopped")
        except Exception as e:
            self.logger.error(f"Error cleaning up cameras: {e}")
    
    def validate_system(self, known_ndvi_targets: List[Tuple[float, Tuple[int, int]]] = None) -> Dict:
        """
        Validate system accuracy against known NDVI targets
        """
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'system_accuracy': 0.0,
            'correlation_coefficient': 0.0,
            'mean_absolute_error': 0.0,
            'target_results': []
        }
        
        if known_ndvi_targets is None:
            self.logger.warning("No validation targets provided")
            return validation_results
        
        try:
            # Capture validation images
            rgb_image, nir_image = self.capture_synchronized_images(save_raw=False)
            
            if rgb_image is None or nir_image is None:
                self.logger.error("Failed to capture validation images")
                return validation_results
            
            # Process images
            result = self.process_image_pair(rgb_image, nir_image)
            
            if not result['success']:
                self.logger.error("Failed to process validation images")
                return validation_results
            
            # Load calculated NDVI
            ndvi_data_path = result['file_paths']['ndvi_data']
            calculated_ndvi = cv2.imread(ndvi_data_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            calculated_ndvi = (calculated_ndvi / 65535.0) * 2 - 1  # Convert back to [-1, 1]
            
            # Compare with known targets
            predicted_values = []
            actual_values = []
            
            for expected_ndvi, (x, y) in known_ndvi_targets:
                # Extract NDVI value at target position
                target_region = calculated_ndvi[y-5:y+5, x-5:x+5]
                predicted_ndvi = np.mean(target_region)
                
                predicted_values.append(predicted_ndvi)
                actual_values.append(expected_ndvi)
                
                validation_results['target_results'].append({
                    'position': (x, y),
                    'expected_ndvi': expected_ndvi,
                    'calculated_ndvi': float(predicted_ndvi),
                    'error': float(abs(predicted_ndvi - expected_ndvi))
                })
            
            # Calculate validation metrics
            if len(predicted_values) > 1:
                correlation = np.corrcoef(actual_values, predicted_values)[0, 1]
                mae = np.mean(np.abs(np.array(predicted_values) - np.array(actual_values)))
                
                validation_results['correlation_coefficient'] = float(correlation)
                validation_results['mean_absolute_error'] = float(mae)
                validation_results['system_accuracy'] = float(correlation * 100)  # Percentage
                
                self.logger.info(f"Validation complete. Accuracy: {correlation*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results

def main():
    """Main function demonstrating system usage"""
    # Initialize NDVI system
    ndvi_system = NDVISystem()
    
    # Example usage
    print("NDVI System Initialized")
    print("Available commands:")
    print("1. Single capture: 's'")
    print("2. Multiple captures: 'm'")
    print("3. Calibration: 'c'")
    print("4. Validation: 'v'")
    print("5. Quit: 'q'")
    
    while True:
        command = input("\nEnter command: ").lower().strip()
        
        if command == 'q':
            break
        elif command == 's':
            # Single capture
            results = ndvi_system.run_capture_session(num_captures=1)
            if results:
                print(f"Capture completed. NDVI mean: {results[0]['ndvi_mean']:.3f}")
        elif command == 'm':
            # Multiple captures
            try:
                num = int(input("Number of captures: "))
                interval = float(input("Interval (seconds): "))
                results = ndvi_system.run_capture_session(num_captures=num, interval=interval)
                print(f"Completed {len(results)} captures")
            except ValueError:
                print("Invalid input. Please enter numbers.")
        elif command == 'c':
            # Calibration
            if ndvi_system.initialize_cameras():
                rgb, nir = ndvi_system.capture_synchronized_images()
                if rgb is not None and nir is not None:
                    success = ndvi_system.perform_calibration(rgb, nir)
                    if success:
                        print("Calibration completed successfully")
                    else:
                        print("Calibration failed")
                ndvi_system.cleanup_cameras()
        elif command == 'v':
            # Validation (example with mock targets)
            print("Running system validation...")
            # Example validation targets (NDVI value, pixel position)
            mock_targets = [
                (0.8, (100, 100)),  # Healthy vegetation
                (0.3, (200, 200)),  # Stressed vegetation
                (0.0, (300, 300))   # Non-vegetation
            ]
            validation = ndvi_system.validate_system(mock_targets)
            print(f"System accuracy: {validation['system_accuracy']:.1f}%")
        else:
            print("Unknown command")
    
    print("NDVI System shutdown")

if __name__ == "__main__":
    main()
