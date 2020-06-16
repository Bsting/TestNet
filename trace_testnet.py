import torch
from testnet import Config, Backbone

if __name__ == '__main__':
    with torch.no_grad():
        conf = Config()
        model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode, conf.net_output)
        model.load_state_dict(torch.load('model.pth', map_location='cpu'))
        model.eval()
        sample = torch.rand(1,3,112,112)
        traced_script_module = torch.jit.trace(model, sample)
        traced_script_module.save("trace_model.pt")
        print(traced_script_module)