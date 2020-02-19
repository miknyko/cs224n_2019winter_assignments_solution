import torch
from nmt_model import Hypothesis, NMT

if __name__ == "__main__":
    model = NMT.load('model.bin', no_char_decoder=True)
    print("[INFO] the model is loaded")
    for i in model.modules():
        print(i)
        print("="*80)
    