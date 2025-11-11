"""
Image Analysis Module - Production Pipeline
===========================================
Complete urinalysis strip analyzer using YOLO + MobileViT.

Pipeline:
1. YOLO detects 11 sensor pads on the strip
2. MobileViT analyzes each pad for urinalysis parameters
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Label Mapping Rules (from training notebook)
# =============================================================================

# 33 main classes
MAIN_CLASSES = sorted([
    'Bilirubin_1', 'Nonhemo_250', 'Nonhemo_10', 'Glucose_2000', 'Nonhemo_50',
    'Glucose_1000', 'Ctrl', 'Glucose_350', 'pH_9', 'Glucose_250', 'Bilirubin_3',
    'Protein_10', 'pH_6.5', 'Glucose_100', 'pH_8', 'pH_6', 'Glucose_500',
    'Nitrite_1', 'Protein_1000', 'Protein_30', 'Hemo_250', 'Protein_300',
    'pH_5', 'Nitrite_0.5', 'pH_7', 'Glucose_1500', 'Hemo_10', 'Glucose_150',
    'Bilirubin_0.5', 'Glucose_750', 'Protein_100', 'pH_7.5', 'Hemo_50'
])

# 6 auxiliary class groups
AUX_CLASSES_GROUPS = {
    'aux_0': sorted(['Hemo_Negative', 'Hemo_10', 'Hemo_50', 'Hemo_250', 
                     'Nonhemo_10', 'Nonhemo_50', 'Nonhemo_250']),
    'aux_1': sorted(['Bilirubin_Negative', 'Bilirubin_0.5', 'Bilirubin_1', 'Bilirubin_3']),
    'aux_4': sorted(['Protein_Negative', 'Protein_10', 'Protein_30', 
                     'Protein_100', 'Protein_300', 'Protein_1000']),
    'aux_5': sorted(['Nitrite_Negative', 'Nitrite_0.5', 'Nitrite_1']),
    'aux_6': sorted(['Glucose_Negative', 'Glucose_100', 'Glucose_150', 'Glucose_250',
                     'Glucose_350', 'Glucose_500', 'Glucose_750', 'Glucose_1000',
                     'Glucose_1500', 'Glucose_2000']),
    'aux_7': sorted(['pH_5', 'pH_6', 'pH_6.5', 'pH_7', 'pH_7.5', 'pH_8', 'pH_9'])
}


# =============================================================================
# YOLO Pad Detector
# =============================================================================

class PadDetector:
    """
    YOLO-based pad detection for urinalysis strips.
    
    Detects 11 sensor pads on a urinalysis strip and returns their coordinates.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLO pad detector.
        
        Args:
            model_path: Path to YOLO weights (.pt file)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        
        if model_path is None:
            model_path = Path(__file__).parent / "yolo.pt"
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.model_path}. "
                "Please provide valid model_path or place yolo.pt in image_analysis/"
            )
        
        print(f"Loading YOLO pad detector from {self.model_path.name}...")
        self.model = YOLO(str(self.model_path))
        print("✓ YOLO pad detector ready!")
    
    def detect_pads(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """
        Detect sensor pads on urinalysis strip.
        
        Args:
            image_path: Path to strip image
            conf_threshold: Confidence threshold for detection
            
        Returns:
            List of detected pads with bounding boxes and class info
            Format: [{'cls': int, 'conf': float, 'bbox': [x1, y1, x2, y2]}, ...]
        """
        # Run YOLO detection
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=0.5,
            max_det=300,
            verbose=False
        )
        
        if len(results) == 0 or results[0].boxes is None:
            return []
        
        result = results[0]
        detections = []
        
        # Extract detection info
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Get absolute coordinates (x1, y1, x2, y2)
            bbox = box.xyxy[0].cpu().numpy().tolist()
            
            # Get normalized coordinates for reference
            bbox_norm = box.xywhn[0].cpu().numpy().tolist()
            
            detections.append({
                'cls': cls_id,
                'conf': confidence,
                'bbox': bbox,  # [x1, y1, x2, y2] in pixels
                'bbox_norm': bbox_norm,  # [x_center, y_center, width, height] normalized
            })
        
        return detections
    
    def extract_11_pads(self, image_path: str, detections: List[Dict]) -> List[np.ndarray]:
        """
        Extract 11 sensor pad patches from image using detections.
        
        Args:
            image_path: Path to original image
            detections: List of detection dictionaries
            
        Returns:
            List of 11 pad images as numpy arrays
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Filter for test_pad class (cls=1)
        pad_detections = [d for d in detections if d['cls'] == 1]
        
        if len(pad_detections) < 11:
            print(f"Warning: Only detected {len(pad_detections)} pads, expected 11")
        
        # Sort pads by Y coordinate (top to bottom)
        pad_detections_sorted = sorted(pad_detections, key=lambda x: x['bbox'][1])
        
        # Take first 11 pads
        selected_pads = pad_detections_sorted[:11]
        
        # Extract pad images
        pad_images = []
        for pad in selected_pads:
            x1, y1, x2, y2 = [int(coord) for coord in pad['bbox']]
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Crop pad region
            pad_img = image[y1:y2, x1:x2]
            pad_images.append(pad_img)
        
        # If less than 11, pad with blank images
        while len(pad_images) < 11:
            blank = np.zeros((50, 50, 3), dtype=np.uint8)
            pad_images.append(blank)
            print(f"Warning: Adding blank pad {len(pad_images)}/11")
        
        return pad_images[:11]


# =============================================================================
# MobileViT Multi-Task Classifier
# =============================================================================

class SensorSetClassifier(nn.Module):
    """
    Multi-Task Learning Urinalysis Classifier
    
    Architecture:
    - 11 specialist expert backbones (MobileViT-XS)
    - 6 auxiliary classifiers for specific parameters
    - 1 main classifier with attention fusion (33 classes)
    """
    
    def __init__(self, 
                 model_name: str = 'mobilevit_xs',
                 feature_dim: int = 384,
                 num_classes_main: int = 33,
                 aux_classes_groups: dict = None,
                 num_sensors: int = 11,
                 nhead: int = 8,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 pretrained: bool = False):
        
        super().__init__()
        
        if aux_classes_groups is None:
            aux_classes_groups = AUX_CLASSES_GROUPS
            
        self.num_classes_main = num_classes_main
        self.aux_classes_groups = aux_classes_groups
        self.num_sensors = num_sensors
        self.feature_dim = feature_dim
        
        # 11 specialist backbones
        print(f"  [Specialist Experts] Creating {num_sensors} MobileViT backbones...")
        self.backbones = nn.ModuleList([
            timm.create_model(model_name, pretrained=pretrained) 
            for _ in range(num_sensors)
        ])
        
        # Verify feature dimension
        actual_features = self.backbones[0].num_features
        if actual_features != self.feature_dim:
            print(f"    Feature dim adjustment: {self.feature_dim} → {actual_features}")
            self.feature_dim = actual_features
        
        for backbone in self.backbones:
            backbone.reset_classifier(0)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention fusion
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Main classifier
        self.main_classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.num_classes_main)
        )
        
        # Auxiliary classifiers
        print(f"  [Auxiliary Heads] Creating 6 specialist classifiers...")
        self.aux_heads = nn.ModuleDict()
        
        for group_name, classes_list in self.aux_classes_groups.items():
            num_aux_classes = len(classes_list)
            pad_index = int(group_name.split('_')[-1])
            
            head = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(self.feature_dim // 2),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim // 2, num_aux_classes)
            )
            
            self.aux_heads[group_name] = head
            print(f"    - {group_name} (pad {pad_index}) → {num_aux_classes} classes")
    
    def forward(self, x):
        """Forward pass"""
        B, N, C, H, W = x.shape
        
        # Extract features from 11 specialists
        patch_features = []
        for i in range(self.num_sensors):
            patch_input = x[:, i, :, :, :]
            features = self.backbones[i].forward_features(patch_input)
            pooled = self.pool(features).flatten(1)
            patch_features.append(pooled)
        
        seq_features = torch.stack(patch_features, dim=1)
        
        # Auxiliary outputs
        aux_outputs = {}
        for group_name, head_mlp in self.aux_heads.items():
            pad_index = int(group_name.split('_')[-1])
            expert_feature = seq_features[:, pad_index, :]
            aux_outputs[group_name] = head_mlp(expert_feature)
        
        # Main output (attention fusion)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq_with_tokens = torch.cat((cls_tokens, seq_features), dim=1)
        contextual_features = self.transformer_encoder(seq_with_tokens)
        cls_output = contextual_features[:, 0]
        main_output = self.main_classifier(cls_output)
        
        return main_output, aux_outputs


# =============================================================================
# Complete Image Analyzer
# =============================================================================

class ImageAnalyzer:
    """
    Complete urinalysis strip analyzer.
    
    Two-stage pipeline:
    1. YOLO detects 11 sensor pads
    2. MobileViT analyzes each pad for urinalysis parameters
    """
    
    def __init__(self, 
                 yolo_path: Optional[str] = None,
                 classifier_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize complete analyzer.
        
        Args:
            yolo_path: Path to YOLO weights (.pt)
            classifier_path: Path to MobileViT weights (.pth)
            device: Device for inference ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print("=" * 60)
        print("Initializing Complete Urinalysis Analyzer")
        print("=" * 60)
        print(f"\nDevice: {self.device}\n")
        
        # Stage 1: Initialize YOLO pad detector
        print("[Stage 1] YOLO Pad Detection")
        print("-" * 60)
        self.pad_detector = PadDetector(model_path=yolo_path)
        
        # Stage 2: Initialize MobileViT classifier
        print("\n[Stage 2] MobileViT Classification")
        print("-" * 60)
        
        if classifier_path is None:
            classifier_path = Path(__file__).parent / "analyzer.pth"
        
        self.classifier_path = Path(classifier_path)
        
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier weights not found at {self.classifier_path}"
            )
        
        # Initialize model architecture
        print(f"Loading classifier from {self.classifier_path.name}...")
        self.classifier = SensorSetClassifier(
            model_name='mobilevit_xs',
            feature_dim=384,
            num_classes_main=len(MAIN_CLASSES),
            aux_classes_groups=AUX_CLASSES_GROUPS,
            pretrained=False
        )
        
        # Load weights
        checkpoint = torch.load(
            self.classifier_path,
            map_location=self.device,
            weights_only=False
        )
        
        if 'model_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            val_acc = checkpoint.get('val_acc', 'unknown')
            print(f"  Model loaded: epoch {epoch}, val_acc {val_acc}%")
        else:
            self.classifier.load_state_dict(checkpoint)
            print(f"  Model loaded successfully")
        
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # Image preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        
        print("\n" + "=" * 60)
        print("✓ Complete Analyzer Ready!")
        print("=" * 60 + "\n")
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze urinalysis strip image.
        
        Pipeline:
        1. YOLO detects pads
        2. Extract pad images
        3. MobileViT classifies
        4. Interpret results
        
        Args:
            image_path: Path to strip image
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing: {image_path}")
        
        # Stage 1: Detect pads with YOLO
        print("  [1/4] Detecting pads...")
        detections = self.pad_detector.detect_pads(image_path)
        pad_count = len([d for d in detections if d['cls'] == 1])
        print(f"        Found {pad_count} pads")
        
        # Stage 2: Extract pad images
        print("  [2/4] Extracting pad regions...")
        pad_images = self.pad_detector.extract_11_pads(image_path, detections)
        print(f"        Extracted {len(pad_images)} pad images")
        
        # Stage 3: Prepare for classification
        print("  [3/4] Preparing for classification...")
        transformed_pads = []
        for pad_img in pad_images:
            transformed = self.transform(pad_img)
            transformed_pads.append(transformed)
        
        sensor_set = torch.stack(transformed_pads).unsqueeze(0).to(self.device)
        
        # Stage 4: Classify with MobileViT
        print("  [4/4] Classifying...")
        with torch.no_grad():
            main_output, aux_outputs = self.classifier(sensor_set)
            
            # Get predictions
            main_probs = torch.softmax(main_output, dim=1)
            main_confidence, main_pred_idx = torch.max(main_probs, dim=1)
            
            main_class = MAIN_CLASSES[main_pred_idx.item()]
            confidence = main_confidence.item()
            
            # Get auxiliary predictions
            aux_predictions = {}
            for group_name, logits in aux_outputs.items():
                probs = torch.softmax(logits, dim=1)
                _, pred_idx = torch.max(probs, dim=1)
                class_list = AUX_CLASSES_GROUPS[group_name]
                aux_predictions[group_name] = class_list[pred_idx.item()]
        
        # Interpret predictions
        results = self._interpret_predictions(main_class, aux_predictions, confidence)
        
        # Add metadata
        results['metadata'] = {
            "model_version": "yolo_mobilevit_mtl",
            "yolo_detections": len(detections),
            "pads_extracted": len(pad_images),
            "device": str(self.device),
            "auxiliary_predictions": aux_predictions,
            "image_path": image_path
        }
        
        print(f"\n  ✓ Analysis complete!")
        print(f"    Main class: {main_class} (confidence: {confidence:.1%})")
        print(f"    UTI probability: {results['UTI_probability']:.1%}\n")
        
        return results
    
    def _interpret_predictions(self, 
                               main_class: str,
                               aux_predictions: Dict[str, str],
                               confidence: float) -> Dict[str, Any]:
        """Convert model predictions to clinical values"""
        results = {
            "main_class": main_class,
            "confidence": confidence
        }
        
        # Extract clinical values from auxiliary predictions
        
        # Hemoglobin (aux_0)
        hemo_pred = aux_predictions.get('aux_0', 'Hemo_Negative')
        if 'Negative' in hemo_pred:
            results['hemoglobin'] = 0.0
        else:
            try:
                results['hemoglobin'] = float(hemo_pred.split('_')[-1])
            except:
                results['hemoglobin'] = 0.0
        
        # Bilirubin (aux_1)
        bili_pred = aux_predictions.get('aux_1', 'Bilirubin_Negative')
        if 'Negative' in bili_pred:
            results['bilirubin'] = 0.0
        else:
            try:
                results['bilirubin'] = float(bili_pred.split('_')[-1])
            except:
                results['bilirubin'] = 0.0
        
        # Protein (aux_4)
        protein_pred = aux_predictions.get('aux_4', 'Protein_Negative')
        if 'Negative' in protein_pred:
            results['protein'] = 0.0
        else:
            try:
                results['protein'] = float(protein_pred.split('_')[-1])
            except:
                results['protein'] = 0.0
        
        # Nitrite (aux_5)
        nitrite_pred = aux_predictions.get('aux_5', 'Nitrite_Negative')
        if 'Negative' in nitrite_pred:
            results['nitrite'] = 0.0
        else:
            try:
                results['nitrite'] = float(nitrite_pred.split('_')[-1])
            except:
                results['nitrite'] = 0.0
        
        # Glucose (aux_6)
        glucose_pred = aux_predictions.get('aux_6', 'Glucose_Negative')
        if 'Negative' in glucose_pred:
            results['glucose'] = 0.0
        else:
            try:
                results['glucose'] = float(glucose_pred.split('_')[-1])
            except:
                results['glucose'] = 0.0
        
        # pH (aux_7)
        ph_pred = aux_predictions.get('aux_7', 'pH_7')
        try:
            results['pH'] = float(ph_pred.split('_')[-1])
        except:
            results['pH'] = 7.0
        
        # Calculate UTI probability
        uti_score = 0.0
        if results['nitrite'] > 0:
            uti_score += 0.4
        if results.get('hemoglobin', 0) > 25:
            uti_score += 0.3
        if results['pH'] > 7.5:
            uti_score += 0.2
        if results['protein'] > 30:
            uti_score += 0.1
        
        results['UTI_probability'] = min(uti_score, 1.0)
        results['lymphocyte'] = 0.0  # Placeholder
        
        return results
    
    def batch_analyze(self, image_paths: List[str]) -> List[Dict]:
        """Analyze multiple images"""
        results = []
        for path in image_paths:
            try:
                result = self.analyze(path)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {path}: {e}")
                results.append({"error": str(e), "image_path": path})
        return results


# Convenience function
def load_analyzer(yolo_path: Optional[str] = None,
                  classifier_path: Optional[str] = None,
                  device: Optional[str] = None) -> ImageAnalyzer:
    """Load complete analyzer"""
    return ImageAnalyzer(
        yolo_path=yolo_path,
        classifier_path=classifier_path,
        device=device
    )


if __name__ == "__main__":
    print("\nTesting Complete Analyzer Pipeline\n")
    try:
        analyzer = ImageAnalyzer()
        print("\n✓ Analyzer ready for inference!")
        print("\nUsage:")
        print("  from image_analysis import ImageAnalyzer")
        print("  analyzer = ImageAnalyzer()")
        print("  results = analyzer.analyze('strip_image.jpg')")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
