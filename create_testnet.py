import torch
from testnet import Config, Backbone

if __name__ == '__main__':
    with torch.no_grad():
        conf = Config()
        model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode, conf.net_output)
        model.eval()
        torch.save(model.state_dict(), 'model.pth')