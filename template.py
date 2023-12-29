def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'
    if args.template.find('test') >= 0:
        args.test_only = 'True'
        args.save_gt = 'True'

    if args.template.find('MSDSN') >= 0:
        args.model = 'MSDSN'
        args.n_resgroups = 8
        args.n_resblocks = 9
        args.n_feats = 64
        args.chop = True
        args.batch_size = 16


