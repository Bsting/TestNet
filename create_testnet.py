import torch
from testnet import Config, Backbone

if __name__ == '__main__':
    conf = Config()
    sample = torch.rand(2,3,112,112)
    model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode, conf.net_output)
    model(sample) # training
    with torch.no_grad():
        model.eval()
        torch.save(model.state_dict(), 'model.pth')