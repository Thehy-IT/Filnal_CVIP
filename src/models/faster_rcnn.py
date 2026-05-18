import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_detection_model(num_classes: int = 2):
    """
    Khởi tạo mô hình Faster R-CNN với ResNet50-FPN backbone.
    Args:
        num_classes: Số lượng class (Background = 0, Lesion = 1 => Tổng là 2)
    """
    # Load pre-trained model (Sử dụng Weights API mới của PyTorch)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    
    # Lấy số lượng features của lớp đầu vào cho classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Thay thế phần head dự đoán bằng một head mới phù hợp với số class của ta
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model