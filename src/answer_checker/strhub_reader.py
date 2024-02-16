from torch import conv2d
import torch
from torchvision import transforms as T
from PIL import Image
from strhub.data.module import SceneTextDataModule
import cv2
import numpy as np

class StrhubReader(object):
    def __init__(self, model_name='parseq') -> None:
        r"""Initialize reader 
        
        Args:
            model_name (str):
                Choose from 'parseq', 'parseq_tiny'
        
        .. notes::
            See also https://github.com/baudm/parseq
        
        """
        self.parseq = torch.hub.load('baudm/parseq', model_name, pretrained=True).eval()
        self.img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        pass
    
    def read_image(self, path_to_image: str):
        img = Image.open(path_to_image).convert('RGB')
        # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
        img = self.img_transform(img).unsqueeze(0)
        return 
    
    def read_text(self, img: torch.Tensor):
        r"""Read text from image patch. Note that patch is first converted to 
        size of (W, H) = (128, 32). Beware of the input dimensions.
        
        Args:
            img (torch.Tensor):
                This should be an RGB image with shape (B, C, H, W)
        
        """
        img = self._preprocess(img).unsqueeze(0)
        img = img.to(self.parseq.device)

        # Greedy decoding
        pred = self.parseq(img).softmax(-1)
        label, _ = self.parseq.tokenizer.decode(pred)
        raw_label, raw_confidence = self.parseq.tokenizer.decode(pred, raw=True)
        # Format confidence values
        max_len = len(label[0]) + 1
        conf = list(map('{:0.1f}'.format, raw_confidence[0][:max_len].tolist()))
        return label[0], [raw_label[0][:max_len], conf]

    @staticmethod
    def cvmat2pil(img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
    def cuda(self):
        self.parseq = self.parseq.cuda()
        self.cuda = True