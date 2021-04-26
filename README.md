要运行代码需要安装以下库：
matplotlib==3.3.2
numpy==1.19.1
cvxpy==1.1.5
torch==1.1.0
torchvision==0.3.0
PyYAML==5.3.1
tensorboardX==2.1

例如： pip install matplotlib==3.3.2
安装torch最好在https://download.pytorch.org/whl/torch_stable.html下自行下载需要的安装包。

安装cvxpy前需安装mkl、cvxopt、scs、ecos、osqp，若安装scs时报错请自行安装
Microsoft visual C++ 14.0，这里提供一个安装包https://blog.csdn.net/weixin_45153285/article/details/90695719


大部分的实验配置都在platform-config.yaml文件中，用户成本和受欢迎程度sij的配置则在Environment.py文件中，如果有需要可以自行更改。

你可以通过运行以下命令开启训练。
python train.py