import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np

# 创建模型对象并加载状态字典
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 修改最后一层为二分类
model.load_state_dict(torch.load('./model/model_61.6.pt', weights_only=True))

# 根据模型所在设备移动模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # 设置为评估模式

def convert_to_cv2(image):
    """将PIL图像转换为OpenCV格式"""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def enhance_saturation(image):
    """增强图像的饱和度"""
    img_cv = convert_to_cv2(image)
    hsv_img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hsv_img_cv[..., 1] = np.clip(hsv_img_cv[..., 1] + 50, 0, 255)  # 限制饱和度在合理范围
    saturated_img_cv = cv2.cvtColor(hsv_img_cv, cv2.COLOR_HSV2BGR)
    return Image.fromarray(cv2.cvtColor(saturated_img_cv, cv2.COLOR_BGR2RGB))

def enhance_contrast(image):
    """增强图像的对比度"""
    img_cv = convert_to_cv2(image)
    enhanced_img_cv = cv2.convertScaleAbs(img_cv, alpha=1.5, beta=0)
    return Image.fromarray(cv2.cvtColor(enhanced_img_cv, cv2.COLOR_BGR2RGB))

def detect_edges(image):
    """检测图像的边缘"""
    img_cv = convert_to_cv2(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # 转换为三通道图像
    return Image.fromarray(edges_3channel)

def histogram_equalization(image):
    """进行直方图均衡化处理"""
    img_cv = convert_to_cv2(image)
    for i in range(3):
        img_cv[:, :, i] = cv2.equalizeHist(img_cv[:, :, i])  # 对每个通道进行均衡化
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def pixelwise_difference(original_img, processed_img):
    """计算原始图像与处理后图像的像素差异"""
    original_np = np.array(original_img)
    processed_np = np.array(processed_img)
    abs_diff = np.abs(original_np - processed_np)
    return np.mean(abs_diff)  # 返回平均绝对差

def extract_sift_features(image):
    """提取图像的SIFT特征"""
    img_cv = convert_to_cv2(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def compare_sift_features(original_descriptors, processed_descriptors):
    """比较两个图像的SIFT特征"""
    if original_descriptors is None or processed_descriptors is None:
        return 0  # 如果特征不存在，返回0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(original_descriptors, processed_descriptors, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]  # 选择好的匹配
    return len(good_matches)

def analyze_differences(original_img, processed_imgs, pixel_threshold=50, sift_threshold=50):
    """分析原始图像与处理后图像之间的差异"""
    differences = []
    for processed_img in processed_imgs:
        pixel_diff = pixelwise_difference(original_img, processed_img)
        sift_features_original = extract_sift_features(original_img)
        sift_features_processed = extract_sift_features(processed_img)
        sift_match_count = compare_sift_features(sift_features_original, sift_features_processed)
        is_suspicious = pixel_diff > pixel_threshold or (sift_match_count < sift_threshold)
        differences.append((pixel_diff, sift_match_count, is_suspicious))
    return differences

def predict_deepfake_probability(image_path):
    """预测图像为Deepfake的概率"""
    original_img = Image.open(image_path).convert('RGB')

    # 进行预处理操作
    sharpened_img = Image.fromarray(cv2.filter2D(np.array(original_img), -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])))
    saturated_img = enhance_saturation(original_img)
    contrast_enhanced_img = enhance_contrast(original_img)
    edges_img = detect_edges(original_img)
    histogram_equalized_img = histogram_equalization(original_img)

    # 定义图像预处理变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 将图像转为张量并移动到设备上
    original_input_tensor = transform(original_img).unsqueeze(0).to(device)
    processed_tensors = {
        "original": original_input_tensor,
        "sharpened": transform(sharpened_img).unsqueeze(0).to(device),
        "saturated": transform(saturated_img).unsqueeze(0).to(device),
        "contrast_enhanced": transform(contrast_enhanced_img).unsqueeze(0).to(device),
        "edges": transform(edges_img).unsqueeze(0).to(device),
        "histogram_equalized": transform(histogram_equalized_img).unsqueeze(0).to(device),
    }

    # 进行预测
    probabilities = {}
    with torch.no_grad():
        for key, tensor in processed_tensors.items():
            output = model(tensor)
            probabilities[key] = output[0][1].item()  # 获取为Deepfake的概率

    # 返回处理后的概率和图像
    return {key: prob / 5 for key, prob in probabilities.items()}, original_img

def analyze_image(probabilities, original_img, processed_imgs):
    """分析图像的Deepfake可能性和处理效果"""
    analysis_results = []

    # 输出Deepfake可能性分析
    probability = probabilities["original"]
    if probability < 0.5:
        analysis_results.append("这张图片看起来很真实。")
        if probability > 0.3:
            analysis_results.append("不过，它可能与一些真实图片有细微的不同。")
    else:
        analysis_results.append("这张图片可能是伪造的。")
        if probability < 0.7:
            analysis_results.append("可能有一些不太明显的伪造迹象，比如颜色看起来不自然。")
        else:
            analysis_results.append("可能有明显的伪造迹象，比如人物的特征不连贯。")

    # 分析不同处理方式后的变化
    for key in probabilities:
        if key != "original":
            if abs(probability - probabilities[key]) > 0.2:
                analysis_results.append(f"经过{key}处理后，这张图片的真实感发生了很大变化，可能有问题。")

    # 像素级对比和特征对比
    for img in processed_imgs:
        diff = pixelwise_difference(original_img, img)
        sift_features = extract_sift_features(img)
        sift_comparison = compare_sift_features(extract_sift_features(original_img), sift_features)
        analysis_results.append(f"与原图相比，{img}的颜色和细节差异为：{diff}。")
        analysis_results.append(f"{img}和原图的特征相似度为：{sift_comparison}。")

    # 分析差异并标记可疑区域
    differences = analyze_differences(original_img, processed_imgs)
    for i, (pixel_diff, sift_count, is_suspicious) in enumerate(differences):
        if is_suspicious:
            analysis_results.append(f"经过处理方法 {i + 1} 后，这张图片可能有可疑的地方。颜色和细节差异为：{pixel_diff}，特征匹配数为：{sift_count}。")
        else:
            analysis_results.append(f"经过处理方法 {i + 1} 后，这张图片看起来没有可疑的地方。颜色和细节差异为：{pixel_diff}，特征匹配数为：{sift_count}。")

    return "\n".join(analysis_results)  # 返回字符串汇总结果


def perform_analysis(image_path):
    # 加载原始图像
    original_img = Image.open(image_path).convert('RGB')
    probabilities, original_img = predict_deepfake_probability(image_path)
    
    # 处理后的图像列表
    processed_imgs = [
        enhance_saturation(original_img),
        enhance_contrast(original_img),
        detect_edges(original_img),
        histogram_equalization(original_img),
    ]
    
    # 调用新的函数进行分析
    analysis_results = analyze_image(probabilities, original_img, processed_imgs)
    
    # 返回分析结果
    return analysis_results

