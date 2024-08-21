import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建模型对象并加载状态字典
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('C:\\Users\\予清\\Desktop\\model_61.6.pt', weights_only=True))

# 根据模型所在设备移动模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def enhance_saturation(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv_img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hsv_img_cv[...,1] += 50
    saturated_img_cv = cv2.cvtColor(hsv_img_cv, cv2.COLOR_HSV2BGR)
    return Image.fromarray(cv2.cvtColor(saturated_img_cv, cv2.COLOR_BGR2RGB))

def enhance_contrast(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    alpha = 1.5
    beta = 0
    enhanced_img_cv = cv2.convertScaleAbs(img_cv, alpha=alpha, beta=beta)
    return Image.fromarray(cv2.cvtColor(enhanced_img_cv, cv2.COLOR_BGR2RGB))

def detect_edges(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    # 将单通道边缘图像转换为三通道图像
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_3channel)

def histogram_equalization(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i in range(3):
        img_cv[:, :, i] = cv2.equalizeHist(img_cv[:, :, i])
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def pixelwise_difference(original_img, processed_img):
    original_np = np.array(original_img)
    processed_np = np.array(processed_img)
    abs_diff = np.abs(original_np - processed_np)
    mean_abs_diff = np.mean(abs_diff)
    return mean_abs_diff

def extract_sift_features(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def compare_sift_features(original_descriptors, processed_descriptors):
    if original_descriptors is None or processed_descriptors is None:
        return None
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(original_descriptors, processed_descriptors, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return len(good_matches)

def analyze_differences(original_img, processed_imgs, pixel_threshold=50, sift_threshold=50):
    differences = []
    original_np = np.array(original_img)
    for processed_img in processed_imgs:
        processed_np = np.array(processed_img)
        pixel_diff = pixelwise_difference(original_img, processed_img)
        sift_features_original = extract_sift_features(original_img)
        sift_features_processed = extract_sift_features(processed_img)
        sift_match_count = compare_sift_features(sift_features_original, sift_features_processed)
        is_suspicious = False
        if pixel_diff > pixel_threshold or (sift_match_count is not None and sift_match_count < sift_threshold):
            is_suspicious = True
        differences.append((pixel_diff, sift_match_count, is_suspicious))
    return differences

def predict_deepfake_probability():
    image_path = input("请输入图片路径：")
    # 去除可能存在的多余引号
    if image_path.startswith('"') and image_path.endswith('"'):
        image_path = image_path[1:-1]
    # 准备输入数据并移动到与模型相同的设备上
    original_img = Image.open(image_path).convert('RGB')

    # 进行预处理操作
    img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    # 锐化操作（示例，可根据实际情况调整参数）
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_img_cv = cv2.filter2D(img_cv, -1, kernel)
    sharpened_img = Image.fromarray(cv2.cvtColor(sharpened_img_cv, cv2.COLOR_BGR2RGB))
    # 饱和度增强操作
    saturated_img = enhance_saturation(original_img)
    # 对比度增强操作
    contrast_enhanced_img = enhance_contrast(original_img)
    # 边缘检测操作
    edges_img = detect_edges(original_img)
    # 直方图均衡化操作
    histogram_equalized_img = histogram_equalization(original_img)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    original_input_tensor = transform(original_img).unsqueeze(0).to(device)
    sharpened_input_tensor = transform(sharpened_img).unsqueeze(0).to(device)
    saturated_input_tensor = transform(saturated_img).unsqueeze(0).to(device)
    contrast_enhanced_input_tensor = transform(contrast_enhanced_img).unsqueeze(0).to(device)
    edges_input_tensor = transform(edges_img).unsqueeze(0).to(device)
    histogram_equalized_input_tensor = transform(histogram_equalized_img).unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        original_output = model(original_input_tensor)
        probability_deepfake = original_output[0][1].item()
        sharpened_output = model(sharpened_input_tensor)
        sharpened_probability_deepfake = sharpened_output[0][1].item()
        saturated_output = model(saturated_input_tensor)
        saturated_probability_deepfake = saturated_output[0][1].item()
        contrast_enhanced_output = model(contrast_enhanced_input_tensor)
        contrast_enhanced_probability_deepfake = contrast_enhanced_output[0][1].item()
        edges_output = model(edges_input_tensor)
        edges_probability_deepfake = edges_output[0][1].item()
        histogram_equalized_output = model(histogram_equalized_input_tensor)
        histogram_equalized_probability_deepfake = histogram_equalized_output[0][1].item()
    return probability_deepfake / 5, sharpened_probability_deepfake / 5, saturated_probability_deepfake / 5, contrast_enhanced_probability_deepfake / 5, edges_probability_deepfake / 5, histogram_equalized_probability_deepfake / 5, original_img, image_path, sharpened_img, saturated_img, contrast_enhanced_img, edges_img, histogram_equalized_img

def analyze_image(image_path, probability, sharpened_probability, saturated_probability, contrast_enhanced_probability, edges_probability, histogram_equalized_probability, original_img, sharpened_img, saturated_img, contrast_enhanced_img, edges_img, histogram_equalized_img):
    # 这里只是给出非常简单的假设性解释，实际中很难准确指出具体不合理之处
    if probability < 0.5:
        print(f"Deepfake 概率为：{probability}。该图像较可能为真实图像。")
        if probability > 0.3:
            print("可能存在一些细微特征与训练集中的真实图像稍有不同，但整体上较为真实。")
    else:
        print(f"Deepfake 概率为：{probability}。该图像较可能为 Deepfake 图像。")
        if probability < 0.7:
            print("可能存在一些不太明显的伪造痕迹，例如颜色过渡不自然或纹理稍微异常。")
        else:
            print("可能有较为明显的伪造迹象，如人物特征不连贯或背景与前景融合不佳。")

    # 分析锐化后的图像概率差异
    if abs(probability - sharpened_probability) > 0.2:
        print("经过锐化处理后，Deepfake 概率有较大变化，可能存在可疑之处。")

    # 分析饱和度增强后的图像概率差异
    if abs(probability - saturated_probability) > 0.2:
        print("经过饱和度增强处理后，Deepfake 概率有较大变化，可能存在可疑之处。")

    # 分析对比度增强后的图像概率差异
    if abs(probability - contrast_enhanced_probability) > 0.2:
        print("经过对比度增强处理后，Deepfake 概率有较大变化，可能存在可疑之处。")

    # 分析边缘检测后的图像概率差异
    if abs(probability - edges_probability) > 0.2:
        print("经过边缘检测处理后，Deepfake 概率有较大变化，可能存在可疑之处。")

    # 分析直方图均衡化后的图像概率差异
    if abs(probability - histogram_equalized_probability) > 0.2:
        print("经过直方图均衡化处理后，Deepfake 概率有较大变化，可能存在可疑之处。")

    # 像素级对比
    sharpened_diff = pixelwise_difference(original_img, sharpened_img)
    saturated_diff = pixelwise_difference(original_img, saturated_img)
    contrast_enhanced_diff = pixelwise_difference(original_img, contrast_enhanced_img)
    edges_diff = pixelwise_difference(original_img, edges_img)
    histogram_equalized_diff = pixelwise_difference(original_img, histogram_equalized_img)

    print(f"锐化后的像素级平均绝对差值：{sharpened_diff}")
    print(f"饱和度增强后的像素级平均绝对差值：{saturated_diff}")
    print(f"对比度增强后的像素级平均绝对差值：{contrast_enhanced_diff}")
    print(f"边缘检测后的像素级平均绝对差值：{edges_diff}")
    print(f"直方图均衡化后的像素级平均绝对差值：{histogram_equalized_diff}")

    # SIFT 特征对比
    original_sift_features = extract_sift_features(original_img)
    sharpened_sift_features = extract_sift_features(sharpened_img)
    saturated_sift_features = extract_sift_features(saturated_img)
    contrast_enhanced_sift_features = extract_sift_features(contrast_enhanced_img)
    edges_sift_features = extract_sift_features(edges_img)
    histogram_equalized_sift_features = extract_sift_features(histogram_equalized_img)

    sharpened_sift_comparison = compare_sift_features(original_sift_features, sharpened_sift_features)
    saturated_sift_comparison = compare_sift_features(original_sift_features, saturated_sift_features)
    contrast_enhanced_sift_comparison = compare_sift_features(original_sift_features, contrast_enhanced_sift_features)
    edges_sift_comparison = compare_sift_features(original_sift_features, edges_sift_features)
    histogram_equalized_sift_comparison = compare_sift_features(original_sift_features, histogram_equalized_sift_features)

    print(f"锐化后的 SIFT 特征匹配数：{sharpened_sift_comparison}")
    print(f"饱和度增强后的 SIFT 特征匹配数：{saturated_sift_comparison}")
    print(f"对比度增强后的 SIFT 特征匹配数：{contrast_enhanced_sift_comparison}")
    print(f"边缘检测后的 SIFT 特征匹配数：{edges_sift_comparison}")
    print(f"直方图均衡化后的 SIFT 特征匹配数：{histogram_equalized_sift_comparison}")

    # 分析差异并标记可疑区域
    processed_imgs = [sharpened_img, saturated_img, contrast_enhanced_img, edges_img, histogram_equalized_img]
    differences = analyze_differences(original_img, processed_imgs)
    for i, (diff_info, img) in enumerate(zip(differences, processed_imgs)):
        pixel_diff, sift_match_count, is_suspicious = diff_info
        if is_suspicious:
            print(f"预处理方法 {i + 1}（可能是锐化、饱和度增强、对比度增强、边缘检测或直方图均衡化）后的图像可能存在可疑区域。像素差异：{pixel_diff}，SIFT 匹配数：{sift_match_count}")
        else:
            print(f"预处理方法 {i + 1}后的图像不太可能存在可疑区域。像素差异：{pixel_diff}，SIFT 匹配数：{sift_match_count}")

if __name__ == "__main__":
    probability, sharpened_probability, saturated_probability, contrast_enhanced_probability, edges_probability, histogram_equalized_probability, original_img, image_path, sharpened_img, saturated_img, contrast_enhanced_img, edges_img, histogram_equalized_img = predict_deepfake_probability()
    analyze_image(image_path, probability, sharpened_probability, saturated_probability, contrast_enhanced_probability, edges_probability, histogram_equalized_probability, original_img, sharpened_img, saturated_img, contrast_enhanced_img, edges_img, histogram_equalized_img)