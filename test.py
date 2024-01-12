import torch
import clip
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def load_model():
    # 加载预训练的CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def calculate_match(image, text, model, preprocess, device, flag):
    # 加载并处理图片
    image = preprocess(image.crop(flag)).unsqueeze(0).to(device)

    # 处理文本
    text = clip.tokenize([text]).to(device)

    # 计算图片和文本的特征向量
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # 计算相似度得分
    similarity = torch.cosine_similarity(image_features, text_features).item()
    return similarity

def pipei(image, text_description, flag):
    model, preprocess, device = load_model()
    score = calculate_match(image, text_description, model, preprocess, device, flag)
    return score

# 加载预训练的物体检测模型
model1 = fasterrcnn_resnet50_fpn(pretrained=True)
model1.eval()

# 图像的预处理
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image)

# 检测图像中的物体
def detect_objects(model, image_tensor):
    with torch.no_grad():
        prediction = model([image_tensor])
    return prediction

# 绘制边界框
def draw_boxes(image, prediction, text_description):
    draw = ImageDraw.Draw(image)
    flag = []
    max = -1
    for element in range(len(prediction[0]['boxes'])):
        boxes = prediction[0]['boxes'][element].cpu().numpy()
        
        score1 = prediction[0]['scores'][element].cpu().numpy()
        if score1 < 0.5:  # 设置一个阈值
            continue

        score = pipei(image, text_description, (boxes[0], boxes[1], boxes[2], boxes[3]))
        if score > max:
            max = score
            flag  = [(boxes[0], boxes[1]), (boxes[2], boxes[3])]
        print(score)
        
    draw.rectangle(flag, outline ="red", width=3)
    return image

# 主函数
def main(image_path, text_description):
    image_tensor = preprocess_image(image_path)
    prediction = detect_objects(model1, image_tensor)
    image = Image.open(image_path)
    result_image = draw_boxes(image, prediction, text_description)
    result_image.show()  # 显示图片
    # result_image.save("output.jpg")  # 或保存图片

# 执行物体检测
if __name__ == "__main__":
    # 示例图片和文本
    image_path = "1.png" # 替换为你的图片文件路径
    text_description = "a women" # 替换为你的描述文本

    main(image_path, text_description)

