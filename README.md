# Installation

## 1. Open new env

```
conda create -n ocr python=3.10
```

Once finished, activate env by `conda activate ocr`

## 2. Install pre-requisit

> Note that this tutorial is for Linux Ubuntu system


### Teserract
```
apt install tesseract
conda install pytersseract
```

### Pytorch
Note that version is important here
```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Strhub parseq

Download sourcecode

```
git clone https://github.com/baudm/parseq.git
```

Make installation requirements.txt

```bash
# Use specific platform build. Other PyTorch 1.13 options: cu116, cu117, rocm5.2
platform=cpu
# Generate requirements files for specified PyTorch platform
make torch-${platform}
# Install the project and core + train + test dependencies. Subsets: [train,test,bench,tune]
pip install -r requirements/core.${platform}.txt -e .[train,test]
```

Note that I use cu117 above, you should enter `make torch-cu117` if you followed the instruction closely.

# Usage

```python
from answer_checker.find_studentid import get_sid

img = cv2.imread("answer_sheet.jpg")
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
reader_names = ['parseq', 'parseq_tiny', 'trba', 'vitstr']
readers = [StrhubReader(name) for name in reader_names]

#! Now you are doing the preprocessing 4 times, what a waste!
res = {}       
for name, reader in zip(reader_names, readers):
    sid, patch = get_sid(img, reader, template) 
    res[name] = sid
```