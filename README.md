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
## 1. Read SID and exam result from scanned images

```python
from answer_checker.find_studentid import get_sid
from answer_checker.exam_result import *

folder = Path("folder/path")

for file in folder.rglob('*jpg'): 
    fname = os.path.join(folderm, file.name)
    img = cv2.imread(fname)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
    reader_names = ['parseq', 'parseq_tiny', 'trba', 'vitstr']
    readers = [StrhubReader(name) for name in reader_names]

    #! Now you are doing the preprocessing 4 times, what a waste!
    #! Get SID from scanned image
    res = {}       
    for name, reader in zip(reader_names, readers):
        sid, patch = get_sid(img, reader, template) 
        res[name] = sid
    
    #! Get answer from scanned image
    all_result = {}
    fail = []
    try: 
        results = get_result(img)
    except Exception as e:
        fail.append(sid)
        continue            
    
    #! Only concat the results with the expected number of answers
    if results: 
        all_result[sid] = results
    else: 
        fail.append(sid)

```

## 2. Export exam results of all scanned images as df/csv

```python


# Format model answer
model_answer_file = r"MCQ answer.txt"
model_answer_alph = get_model_answer(model_answer_file)
model_answer = format_model_answer(model_answer_alph)

# Export ouput to csv
df = create_output(model_answer, all_result)
df.to_csv('MCQ_result.csv')
```