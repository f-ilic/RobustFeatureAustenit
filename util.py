import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from torchvision.models.resnet import resnet18

# CHW tensor input -> CHW tensor output
def add_text(x: torch.Tensor, text: str, position=(0,0), fill=(255,255,255), stroke_fill=(0,0,0), stroke_width=2, fontsize=35) -> torch.Tensor:
    x = ToPILImage()(x)
    draw = ImageDraw.Draw(x)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", fontsize)
    draw.multiline_text(position, text, font=font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_width)
    return ToTensor()(x)

def set_requires_grad(model, flag: bool):
    for param in model.parameters():
        param.requires_grad = flag

def load_weights(model, optimizer, weights_path):
    if weights_path != None:
        pth = torch.load(weights_path)
        model.load_state_dict(pth['state_dict'])
        print(f"Load Model from: {weights_path}")
        model.name = pth['name']

        if optimizer != None:
            print(f"Load Optimizer from: {weights_path}")
            optimizer.load_state_dict(pth['optimizer'])
            # hacky garbage from https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    else:
        print(f"Nothing to Load")

# log the run dictionary that is produced by the do(...) function
def log(mode, rundict, writer, epoch=0, end='\n'):
    string = ""
    for k,v in rundict.items():
        v = v.avg if hasattr(v, 'avg') else v # v might be average meter, but might also just be a basictype
        if writer != None:
            writer.add_scalar(f'{mode}/{k}', v, epoch)
        string += f'{k}: {v:05.2f} | '
    
    print(string, end=end)


# take list of images and concat them to one big image.
def image_to_grid(x: list, num_rows, num_columns):
    """ x = [C,H,W] """
    assert len(x) == num_rows*num_columns
    shape = x[0].shape
    for cx in x:
        assert cx.shape == shape

    rows = [x[i*num_columns:(i+1)*num_columns] for i in range(num_rows) ] 

    rows_cat = [ torch.cat(row, dim=-1) for row in rows]
    img = torch.cat(rows_cat, dim=-2)
    return img


def get_head(in_features, num_classes, head_type):
    # use this for REGRESSION
    if head_type == 'extended':
        head =  nn.Sequential(
            nn.Linear(in_features, 256),
            # nn.Dropout(),  
            nn.ReLU(),
            nn.Linear(256, 64),
            # nn.Dropout(),  
            nn.ReLU(),
            nn.Linear(64, num_classes))
    elif head_type == 'classic':
        # use this for CLASSIFICATOIN
        head = nn.Sequential(nn.Linear(in_features, num_classes))
    else:
        raise ValueError("head type wrong")
    return head
    

def create_model(backbone: str, num_classes: int, freeze_backbone: bool, head_type: str):
    assert backbone in ['resnet18', 'alexnet', 'vgg16']
    assert head_type in ['classic', 'extended']

    if backbone == 'resnet18':
        in_features = 512 
        model = ResNet18Custom(True)
    set_requires_grad(model, flag=not freeze_backbone)
    model.classifier = get_head(in_features, num_classes, head_type)
    return model


class ResNet18Custom(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18Custom, self).__init__()
        resnet = resnet18(pretrained)
        # TODO maybe this might not be optimal as self.features has the adaptivePool2d in there.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(list(resnet.children())[-1])
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
