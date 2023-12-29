import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import utility as utility
import data as data
import model as model
import loss as loss
from option import args
# from option_test import args
from trainer import Trainer
from videotester import VideoTester

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        # from videotester import VideoTester
        model = model.Model(args, checkpoint)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)

            print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                # t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()

