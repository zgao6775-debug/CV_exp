import os
import cv2
from .model import load_model

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 加载模型 (使用 yolov8n.pt，会自动下载)
    model = load_model('yolov8n.pt')
    if model is None:
        return
        
    # COCO 数据集中，'bicycle' 的类别索引通常是 1
    # 我们可以通过 model.names 查看
    target_class_id = 1 # bicycle
    
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Found {len(input_files)} images in {input_dir}")
    
    for filename in input_files:
        print(f"Processing {filename}...")
        img_path = os.path.join(input_dir, filename)
        
        # 运行推理
        # conf=0.25: 置信度阈值
        # save=False: 不使用内置保存，我们自己处理
        results = model(img_path, conf=0.25)
        
        # 处理结果
        for result in results:
            img = result.orig_img.copy()
            boxes = result.boxes
            
            detected_count = 0
            
            for box in boxes:
                # 获取类别ID
                cls_id = int(box.cls[0])
                
                # 只保留自行车 (bicycle)
                # 有些时候单车可能被识别为 motorbike (3)，也可以考虑包含进来
                if cls_id == 1: 
                    detected_count += 1
                    
                    # 获取坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # 绘制矩形框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制标签
                    label = f"Bicycle {conf:.2f}"
                    t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(img, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1, y1 - 2), 0, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            
            print(f"Detected {detected_count} bicycles in {filename}")
            
            # 保存结果
            out_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(out_path, img)

if __name__ == "__main__":
    main()
