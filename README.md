Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild

### Note

* The default dataset is `300W`
* Check `convert` function in `utils/util.py` for processing original dataset

### Installation

```
conda create -n PyTorch python=3.8
conda activate PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install tqdm
```

### Train

* Configure your dataset path in `main.py` for training
* Download pretrained weights, see `Pretrained weights`
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Demo

* Configure your video path in `main.py` for visualizing the demo
* Run `python main.py --demo` for demo

### Results

| Backbone | Epochs | Test NME |                                                                 Pretrained weights |
|:--------:|:------:|---------:|-----------------------------------------------------------------------------------:|
| IRNet18  |  120   |     3.27 |  [model](https://github.com/jahongir7174/PIPNet/releases/download/v0.0.1/IR18.pth) |
| IRNet50  |  120   |     3.11 |  [model](https://github.com/jahongir7174/PIPNet/releases/download/v0.0.1/IR50.pth) |
| IRNet100 |  120   |     3.08 | [model](https://github.com/jahongir7174/PIPNet/releases/download/v0.0.1/IR100.pth) |

![Alt Text](./demo/demo.gif)

#### Reference

* https://github.com/jhb86253817/PIPNet
