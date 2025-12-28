import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from .model import Net
from .preprocess import preprocess_image

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')
    model_path = os.path.join(base_dir, 'models', 'mnist_cnn.pt')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    if not os.path.exists(model_path):
        print("Model file not found. Please run train.py first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 预处理转换 (与训练时一致)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 处理输入图片
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in input_files:
        print(f"Processing {filename}...")
        img_path = os.path.join(input_dir, filename)
        
        try:
            # 分割数字
            digits, original_img = preprocess_image(img_path)
            
            if not digits:
                print(f"No digits found in {filename}")
                continue
                
            results = []
            
            # 识别每个数字
            for i, digit_img in enumerate(digits):
                # 保存分割后的数字图片 (调试用)
                cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_digit_{i}.png"), digit_img)
                
                # 转换格式
                # digit_img 是 numpy array (28, 28), uint8
                # 需要转换为 PIL Image 或者直接转 Tensor
                tensor = transform(digit_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(tensor)
                    pred = output.argmax(dim=1).item()
                    
                    # 打印概率分布 (调试用)
                    probs = torch.exp(output)[0]
                    top3_prob, top3_idx = torch.topk(probs, 3)
                    print(f"Digit {i} prediction: {pred}, Top-3: {[(idx.item(), round(prob.item(), 2)) for idx, prob in zip(top3_idx, top3_prob)]}")
                    
                    results.append(str(pred))
            
            # 输出结果
            student_id = "".join(results)
            print(f"Recognized Student ID: {student_id}")
            
            # 在原图上标记 (简单把结果写在图片上)
            result_img = original_img.copy()
            cv2.putText(result_img, f"ID: {student_id}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.png"), result_img)
            
            # 保存到文本
            with open(os.path.join(output_dir, "result.txt"), "a") as f:
                f.write(f"{filename}: {student_id}\n")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
