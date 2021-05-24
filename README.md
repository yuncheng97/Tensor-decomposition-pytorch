# Tensor-decomposition-pytorch
CP and Tucker decomposition for AlexNet, VGGNet-16 and ResNet-50

This is our course project of Matrix Analysis. 

# Pre-trained models and Pre-decomposed models

- Google Drive: https://drive.google.com/drive/folders/1QdbakxEIXOyD35cSiLJTJZTndPprn0Q0?usp=sharing

  You can use our pre-decomposed model to do the inference directly.

# Usage

- Dataset:
We use the DogvsCat dataset provided by Kaggle competition. you can download in https://www.kaggle.com/c/dogs-vs-cats/overview
- Training:
``python main.py --train --network [args.network] --model_path [args.model_path]``

  The trained model will be saved in './model_path'
- Decomposition:
``python main.py --decompose --model_path [args.model_path] --decompose_model_path [args.decompose_model_path]``

  The decomposed model will be saved in './decompose_model_path'

- Fine-tuning:
``python main.py --fine_tune --decompose_model_path [args.decompose_model_path]``

  it will show the prediction accuracy of decomposed model
  
- Testing:
``python main.py --test``

  Using the trained model to test on unseen images. The result is a csv file which contains the prediction result with the official submission form of DogvsCat competition. It will saved in [args.result] path

- Visualization:
``python main.py --vis``
  
  It will show the feature map extracted from the convolution layer you set
# TO DO

built more network and test the performance after decomposition and fine-tune. Whether the Tucker decomposition works better for larger & more complex model or smaller & simpler model? 
  
# References

- CP Decomposition for convolutional layers is described here: https://arxiv.org/abs/1412.6553
- Tucker Decomposition for convolutional layers is described here: https://arxiv.org/abs/1511.06530
- VBMF for rank selection is described here: http://www.jmlr.org/papers/volume14/nakajima13a/nakajima13a.pdf
- VBMF code was taken from here: https://github.com/CasvandenBogaard/VBMF
- Tensorly: https://github.com/tensorly/tensorly
