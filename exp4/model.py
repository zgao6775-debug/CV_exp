import os
from ultralytics import YOLO

def load_model(model_name='yolov8n.pt'):
    """
    加载YOLOv8模型，如果本地不存在则会自动下载
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    # 确保models目录存在
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # 模型路径
    # 注意：ultralytics 库会自动处理下载，默认下载到当前工作目录或系统缓存
    # 为了方便管理，我们可以先指定一个路径，或者直接让它自己下载
    # 这里我们直接使用 ultralytics 的自动下载功能
    
    print(f"Loading YOLOv8 model: {model_name}...")
    try:
        model = YOLO(model_name)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
