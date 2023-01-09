import os
import json

import torch
from PIL import Image
from torchvision import transforms
from utils.load_model import GetInitResnet34, LoadTrainedWeight, GetResnet34_CBAM
from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # ---------------------- 指向需要遍历预测的图像文件夹 ----------------------
    imgs_root = "./data_set/food_data/test"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, 'img_' + str(i) + '.jpg') for i in range(500)]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    # model = resnet34(num_classes=10).to(device)
    model = GetResnet34_CBAM(10, device)

    # load model weights
    weights_path = "./Result/best_resnet34_cbam.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 1  # 每次预测时将多少张图片打包成一个batch
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            # 写入结果文件
            result_path = r'./result_cbam.txt'
            fp = open(result_path, 'a', encoding='utf-8')

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
                fp.write(str(cla.numpy()) + '\n')
            fp.close()


if __name__ == '__main__':
    main()

# 0 冰激凌
# 1 鸡蛋布丁
# 2 烤冷面
# 3 芒果班戟
# 4 三明治
# 5 松鼠鱼
# 6 甜甜圈
# 7 土豆泥
# 8 小米粥
# 9 玉米饼