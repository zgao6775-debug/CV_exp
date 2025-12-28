import cv2
import numpy as np

def preprocess_image(image_path):
    """
    预处理学号照片：灰度化 -> 二值化 -> 轮廓检测 -> 排序 -> 分割数字
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化 (Adaptive Thresholding 适应光照不均)
    # 块大小必须是奇数，C是常数
    # C值越小，保留的细节越多。之前是10，可能太大了导致细节丢失。
    # 恢复 C=2, blockSize=21 (最稳定的配置)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 2)
    
    # 简单的去噪
    # 改用闭运算连接断裂的笔画
    # 恢复标准 (2, 2) 闭运算，平衡去噪和连接
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 初步筛选
    candidates = []
    h_img, w_img = img.shape[:2]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 极小的噪点去除
        if w < 5 or h < 10: continue
        if w * h < 100: continue
        
        candidates.append((x, y, w, h))
        
    if not candidates:
        return [], img

    # 动态筛选：基于中位数高度
    # 数字通常大小一致
    heights = [c[3] for c in candidates]
    median_h = np.median(heights)
    
    digit_rects = []
    for (x, y, w, h) in candidates:
        # 高度应该在中位数的一定范围内 (0.5 ~ 2.0 倍)
        if h < median_h * 0.5 or h > median_h * 2.5:
            continue
            
        # 处理连笔/粘连字符
        # 如果宽度明显大于高度 (例如 w > 1.0 * h)，可能是两个数字粘连
        # 再次调高阈值到 1.5，避免把宽写的'2'切分
        if w > h * 1.5:
            # 尝试分割
            # 简单分割：直接从中间切开
            # 进阶分割：垂直投影找波谷 (这里用简单的中间切分，因为投影需要二值化后的图)
            
            # 为了更准确，我们还是用投影法
            # 1. 获取ROI
            roi = thresh[y:y+h, x:x+w]
            
            # 2. 垂直投影 (按列求和)
            vertical_projection = np.sum(roi, axis=0) / 255.0
            
            # 3. 在中间区域 (25% - 75%) 找最小值作为分割点
            # 稍微放宽搜索范围
            start_search = int(w * 0.25)
            end_search = int(w * 0.75)
            
            if end_search > start_search:
                search_region = vertical_projection[start_search:end_search]
                min_val_idx = np.argmin(search_region)
                min_val = search_region[min_val_idx]
                
                # 验证波谷深度：
                # 计算整个投影的平均高度 (非零区域)
                non_zero_proj = vertical_projection[vertical_projection > 0]
                if len(non_zero_proj) > 0:
                    avg_proj = np.mean(non_zero_proj)
                else:
                    avg_proj = min_val + 1 # avoid zero division
                
                # 只有当波谷明显低于平均水平 (例如 < 85%) 时才分割
                # 再次降低阈值到 0.60，只有波谷非常深才切分
                if min_val < avg_proj * 0.60:
                    split_idx = start_search + min_val_idx
                    digit_rects.append((x, y, split_idx, h))
                    digit_rects.append((x + split_idx, y, w - split_idx, h))
                else:
                    # 波谷不明显，认为是单个宽字符
                    digit_rects.append((x, y, w, h))
            else:
                # 无法搜索
                digit_rects.append((x, y, w, h))
        else:
            digit_rects.append((x, y, w, h))
        
    # 按x坐标排序 (从左到右)
    digit_rects.sort(key=lambda r: r[0])
    
    # 简单的非极大值抑制 (NMS) 去除重叠框
    # 有时候一个数字会被切成两个框，或者噪点框在数字里面
    final_rects = []
    for r in digit_rects:
        if not final_rects:
            final_rects.append(r)
            continue
            
        last_r = final_rects[-1]
        # 检查重叠
        # 如果当前框的中心在和上一个框重叠严重，或者包含关系
        x1, y1, w1, h1 = last_r
        x2, y2, w2, h2 = r
        
        # 简单的合并逻辑：如果x距离很近，且重叠
        if x2 < x1 + w1 * 0.8: # 重叠超过 20%
            # 合并两个框
            nx = min(x1, x2)
            ny = min(y1, y2)
            nw = max(x1+w1, x2+w2) - nx
            nh = max(y1+h1, y2+h2) - ny
            final_rects[-1] = (nx, ny, nw, nh)
        else:
            final_rects.append(r)
            
    digits = []
    
    # 提取并标准化每个数字
    padding = 10
    for x, y, w, h in final_rects:
        # 裁剪，稍微留点边距
        # 注意边界检查
        y1 = max(0, y - padding)
        y2 = min(h_img, y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(w_img, x + w + padding)
        
        roi = thresh[y1:y2, x1:x2]
        
        # 调整大小到 28x28 (MNIST 标准预处理方式：Center of Mass)
        # 1. 先将数字缩放到 20x20 的框内，保持纵横比
        h_roi, w_roi = roi.shape
        if h_roi > w_roi:
            factor = 20.0 / h_roi
            new_h = 20
            new_w = int(w_roi * factor)
        else:
            factor = 20.0 / w_roi
            new_w = 20
            new_h = int(h_roi * factor)
            
        resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 2. 创建 28x28 画布
        padded_digit = np.zeros((28, 28), dtype=np.uint8)
        
        # 3. 计算质心 (Center of Mass)
        # 这一步非常关键，MNIST是根据质心居中，而不是几何中心
        M = cv2.moments(resized_roi)
        if M["m00"] > 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        else:
            cX, cY = new_w / 2, new_h / 2
            
        # 4. 将数字贴到画布上，使得质心位于 (14, 14)
        # 计算偏移量
        shift_x = 14 - cX
        shift_y = 14 - cY
        
        # 初始贴图位置（几何中心）
        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        
        # 修正后的贴图位置
        # 这里需要精细控制，避免越界
        # 简单做法：先贴到几何中心，然后整体平移
        temp_canvas = np.zeros((28, 28), dtype=np.uint8)
        temp_canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_roi
        
        # 重新计算temp_canvas的质心
        M_temp = cv2.moments(temp_canvas)
        if M_temp["m00"] > 0:
            tcX = M_temp["m10"] / M_temp["m00"]
            tcY = M_temp["m01"] / M_temp["m00"]
            
            dx = int(14 - tcX)
            dy = int(14 - tcY)
            
            # 平移矩阵
            M_affine = np.float32([[1, 0, dx], [0, 1, dy]])
            padded_digit = cv2.warpAffine(temp_canvas, M_affine, (28, 28))
        else:
            padded_digit = temp_canvas
            
        # 再次加粗一点点，适配模型
        # 使用水平方向的膨胀，专门修复 '2' 的底部横线断裂，同时不影响 '0' 的垂直空洞
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        padded_digit = cv2.dilate(padded_digit, kernel_h, iterations=1)
        
        # 再做一个极其轻微的全向膨胀？或者不需要
        
        digits.append(padded_digit)
        
    return digits, img
