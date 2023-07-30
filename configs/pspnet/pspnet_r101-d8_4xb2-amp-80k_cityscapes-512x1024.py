_base_ = './pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py'
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    loss_scale=512.)
