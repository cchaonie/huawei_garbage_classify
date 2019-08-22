import ast
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from model_service.pytorch_model_service import PTServingBaseService

def init_model(model_name, state_path, num_classes):
    model = models.resnet18() if model_name == 'resnet18' else models.resnet50()
    for param in model.parameters():
        param.requires_grad = False
    num_fc_if = model.fc.in_features
    model.fc = nn.Linear(num_fc_if, num_classes)
    model.load_state_dict(torch.load(state_path, map_location='cpu'))
    return model

class garbage_classify_service(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        model = init_model("resnet50", self.model_path, 40)
        self.model = model
        self.model.eval()

        self.label_id_name_dict = \
            {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉",
                "10": "厨余垃圾/茶叶渣",
                "11": "厨余垃圾/菜叶菜根",
                "12": "厨余垃圾/蛋壳",
                "13": "厨余垃圾/鱼骨",
                "14": "可回收物/充电宝",
                "15": "可回收物/包",
                "16": "可回收物/化妆品瓶",
                "17": "可回收物/塑料玩具",
                "18": "可回收物/塑料碗盆",
                "19": "可回收物/塑料衣架",
                "20": "可回收物/快递纸袋",
                "21": "可回收物/插头电线",
                "22": "可回收物/旧衣服",
                "23": "可回收物/易拉罐",
                "24": "可回收物/枕头",
                "25": "可回收物/毛绒玩具",
                "26": "可回收物/洗发水瓶",
                "27": "可回收物/玻璃杯",
                "28": "可回收物/皮鞋",
                "29": "可回收物/砧板",
                "30": "可回收物/纸板箱",
                "31": "可回收物/调料瓶",
                "32": "可回收物/酒瓶",
                "33": "可回收物/金属食品罐",
                "34": "可回收物/锅",
                "35": "可回收物/食用油桶",
                "36": "可回收物/饮料瓶",
                "37": "有害垃圾/干电池",
                "38": "有害垃圾/软膏",
                "39": "有害垃圾/过期药物"
            }

    def preprocess_img(self, img):
        infer_transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        img = infer_transformation(img)
        return img

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            input_batch = []
            for _, file_content in v.items():
                with Image.open(file_content) as image1:
                    image1 = image1.convert("RGB")
                    input_batch.append(self.preprocess_img(image1))
            input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
            preprocessed_data[k] = input_batch_var
        return preprocessed_data

    def _inference(self, data):
        img = data['input_img']
        outputs = self.model(img)
        print('inferred output:', outputs)
        if outputs is not None:
            _, pred_label = torch.max(outputs, 1)
            result = {'result': self.label_id_name_dict[str(pred_label.item())]}
        else:
            result = {'result': 'predict score is None'}
        return result

    def _postprocess(self, data):
        return data
