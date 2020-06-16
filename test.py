import torch
from testnet import Config, Backbone

if __name__ == '__main__':
    with torch.no_grad():
        torch.manual_seed(0)
        sample = torch.rand(1,3,112,112)
        
        print('\n----inference result--------\n')
        conf = Config()
        model_test = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode, conf.net_output)
        model_test.load_state_dict(torch.load('model.pth', map_location='cpu'))
        model_test.eval()
        test_result = model_test(sample)
        print (test_result)
        
        print('\n----trace inference result--------\n')
        model_trace_test = torch.jit.load("trace_model.pt")
        model_trace_test.eval()
        test_trace_result = model_trace_test(sample)
        print (test_trace_result)