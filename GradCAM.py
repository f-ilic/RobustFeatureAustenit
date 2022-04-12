import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

class CamExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.model.classifier(x)
        return conv_output, x

class GradCAM():
    def __init__(self, model, target_layer, img_width, img_height, mean, std):
        self.model = model
        self.model.eval()
        self.mean = mean
        self.std = std

        self.img_height = img_height
        self.img_width = img_width

        self.normalize = transforms.Normalize(self.mean, self.std)
        self.renormalize = transforms.Normalize(mean=[-m / s for m, s in zip(self.mean, self.std)],
                                                std=[1 / s for s in self.std])
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None, normalize=True):
        assert(self.img_height == input_image.shape[1])
        assert(self.img_width == input_image.shape[2])

        # input_image = self.normalize(transforms.ToTensor()(input_image)).unsqueeze(0)
        input_image = self.normalize(input_image).unsqueeze(0)


        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_class] = 1

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()

        model_output.backward(gradient=one_hot_output, retain_graph=True)

        guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]
        target = conv_output.cpu().data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        cam = np.ones(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        if normalize:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS)) / 255
        cam = np.double(Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS))
        return self.renormalize(input_image).squeeze(), torch.Tensor(cam).T