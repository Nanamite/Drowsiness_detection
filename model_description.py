from generate_model import *
from torchinfo import summary

mobile_net = gen_mobile_net(False)
squeeze_net = gen_squeeze_net(False)

fp1 = open(r'model_architecture_description\mobile_net_arch.txt', 'w+', encoding="utf-8")
fp2 = open(r'model_architecture_description\squeeze_net_arch.txt', 'w+', encoding="utf-8")

fp1.write(str(summary(mobile_net)))
fp2.write(str(summary(squeeze_net)))

fp1.close()
fp2.close()