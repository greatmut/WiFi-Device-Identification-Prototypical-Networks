# WiFi Device Identification Using Prototypical Networks

Few-shot learning approach for WiFi device identification using scalogram analysis and prototypical networks.

## Features
- Custom WiFi scalogram dataset support
- Modified ResNet18 backbone for few-shot learning
- Prototypical Networks implementation
- Windows/Linux compatible

## Installation
```bash
git clone https://github.com/greatmut/WiFi-Device-Identification-Prototypical-Networks.git
cd WiFi-Device-Identification-Prototypical-Networks
pip install -r requirements.txt
```

## Usage
```python
python train.py --n_way 5 --n_shot 5 --epochs 20
```

## Dataset Structure
```
dataset/
    background/
        device1/
            image1.png
            image2.png
        device2/
            ...
    evaluation/
        ...
```

## Results
Achieved XX% accuracy on YY-class WiFi device identification task.

## Contributors
- [Mutala Mohammed](https://github.com/greatmut)
- Shanghai Jiao Tong University

## References
Based on original work by Sicara: [Easy Few-Shot Learning](https://github.com/sicara/easy-few-shot-learning)
