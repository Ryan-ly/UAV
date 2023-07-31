_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

# ========================modified parameters======================
deepen_factor = 1.33
widen_factor = 1.25
num_det_layers = 3

data_root = '/datasets/UAV/detect/1133/'
train_ann_file = 'annotations/trainval.json'
train_data_prefix = 'images/'
val_ann_file = 'annotations/test.json'
val_data_prefix = 'images/'
work_dir = './work_dirs/UAV'

num_classes = 7
class_name = (
    'construction',
    'trafficSign',
    'billboard',
    'roadMarking',
    'car',
    'spillage',
    'illegalSign',
)
# metainfo 必须要传给后面的 dataloader 配置，否则无效
# palette 是可视化时候对应类别的显示颜色
# palette 长度必须大于或等于 classes 长度
metainfo = dict(classes=class_name)
img_scale = (640,640)

train_batch_size_per_gpu = 2
train_num_workers = 2

base_lr = 0.01
max_epochs = 50

affine_scale = 0.9
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0

lr_factor = 0.1
weight_decay = 0.0005
save_checkpoint_intervals = 10
save_epoch_intervals = 10

max_keep_ckpts = 3 #保存模型最多几个



model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(
        head_module=dict(widen_factor=widen_factor,num_classes=num_classes),
        loss_cls=dict(loss_weight=loss_cls_weight *
                      (num_classes / 80 * 3 / num_det_layers)),
        loss_obj=dict(loss_weight=loss_obj_weight *
                      ((img_scale[0] / 640)**2 * 3 / num_det_layers))))

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=20,  # 第几个 epoch 后验证，这里设置 20 是因为前 20 个 epoch 精度不高，测试意义不大，故跳过
    val_interval=save_checkpoint_intervals  # 每 val_interval 轮迭代进行一次测试评估
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        metainfo=metainfo
    )
)


val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=train_data_prefix),
        ann_file=train_ann_file,
        metainfo=metainfo)
)

test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

default_hooks = dict(
    # 每隔 10 个 epoch 保存一次权重，并且最多保存 2 个权重
    # 模型评估时候自动保存最佳模型
    checkpoint=dict(
        interval=save_epoch_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'
        ),
    # warmup_mim_iter 参数非常关键，因为 cat 数据集非常小，默认的最小 warmup_mim_iter 是 1000，导致训练过程学习率偏小
    param_scheduler=dict(
        max_epochs=max_epochs,
        warmup_mim_iter=10),
    # 日志打印间隔为 5
    logger=dict(type='LoggerHook', interval=10))
# vis_backends = [dict(type='WandbVisBackend')]

