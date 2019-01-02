from pprint import pprint

from fastai.callbacks import *
from fastai.vision import *
from hyperopt import fmin, tpe, hp, Trials


@dataclass
class MultiTrainSaveModelCallback(SaveModelCallback):
    def on_train_begin(self, **kwargs):
        old_best = self.best if hasattr(self, 'best') else None
        super().on_train_begin(**kwargs)
        self.best = old_best or self.best


@dataclass
class MultiTrainEarlyStoppingCallback(EarlyStoppingCallback):
    def on_train_begin(self, **kwargs):
        old_best = self.best if hasattr(self, 'best') else None
        super().on_train_begin(**kwargs)
        self.best = old_best or self.best

    def on_epoch_end(self, epoch, **kwargs):
        self.early_stopped = super().on_epoch_end(epoch, **kwargs)
        return self.early_stopped


class FocalLoss(nn.Module):
    def __init__(self, gamma, class_weight=None):
        super().__init__()
        self.gamma = gamma
        self.class_weight = class_weight

    def forward(self, logit, target):
        target = target.view(-1, 1).long()

        b, c, h, w = logit.size()

        class_weight = self.class_weight
        if class_weight is None:
            class_weight = [1] * c

        logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, c)
        prob = F.softmax(logit, 1)
        select = torch.FloatTensor(len(prob), c).zero_().cuda()
        select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        return batch_loss.mean()


def resnet_split(model):
    return (model[0][6], model[1])


def create_data(batch_size):
    return (
        ImageItemList
            .from_folder(untar_data(URLs.CIFAR))
            .random_split_by_pct(valid_pct=0.2, seed=42)
            .label_from_folder()
            .transform(get_transforms())
            .add_test_folder('test')
            .databunch(bs=batch_size)
    )


def create_learner(data, model_factory_func, model_split_func, models_base_path, loss_func):
    return create_cnn(
        data,
        model_factory_func,
        pretrained=True,
        split_on=model_split_func,
        metrics=[accuracy],
        path=models_base_path,
        loss_func=loss_func
    )


def bootstrap_training():
    model_name = 'resnet34'
    models_base_path = Path('/artifacts')

    data = create_data(batch_size=64)
    learn = create_learner(data, models.resnet34, resnet_split, models_base_path, nn.CrossEntropyLoss())

    model_saving = MultiTrainSaveModelCallback(learn, monitor='accuracy', mode='max', name=model_name)
    early_stopping = MultiTrainEarlyStoppingCallback(learn, monitor='accuracy', mode='max', patience=1, min_delta=1e-3)

    learn.callbacks = [model_saving, early_stopping]

    freeze_lr = 1e-2
    unfreeze_lr = 1e-3

    learn.freeze()
    early_stopping.patience = 3
    learn.fit(100, lr=freeze_lr)

    learn.unfreeze()
    early_stopping.patience = 3
    learn.fit(100, lr=slice(unfreeze_lr))

    cycle_len = 10
    early_stopping.patience = cycle_len - 1
    early_stopping.early_stopped = False
    while not early_stopping.early_stopped:
        learn.fit_one_cycle(cycle_len, max_lr=unfreeze_lr)


def get_loss_func(loss_config):
    if loss_config['type'] == 'cce':
        return nn.CrossEntropyLoss()
    elif loss_config['type'] == 'focal':
        return FocalLoss(gamma=loss_config['gamma'])
    else:
        raise Exception(f'Unsupported loss type "{loss_config["type"]}"')


def train(args):
    pprint(args)
    print(flush=True)

    model_name = 'resnet34'
    loss_config = args[0]

    loss_func = get_loss_func(loss_config)

    models_base_path = Path('/artifacts')

    if not os.path.isfile(f'{models_base_path}/models/{model_name}.pth'):
        bootstrap_training()

    data = create_data(batch_size=64)
    learn = create_learner(data, models.resnet34, resnet_split, models_base_path, loss_func)

    model_saving = MultiTrainSaveModelCallback(learn, monitor='accuracy', mode='max', name=model_name)
    early_stopping = MultiTrainEarlyStoppingCallback(learn, monitor='accuracy', mode='max', patience=1, min_delta=1e-3)

    learn.callbacks = [model_saving, early_stopping]

    learn.load(model_name)

    unfreeze_lr = 1e-3
    cycle_len = 10
    early_stopping.patience = cycle_len - 1
    early_stopping.early_stopped = False
    while not early_stopping.early_stopped:
        learn.fit_one_cycle(cycle_len, max_lr=unfreeze_lr)

    return -early_stopping.best


hyper_space = [
    hp.choice('loss', (
        {
            'type': 'cce'
        },
        {
            'type': 'focal',
            'gamma': hp.choice('focal_loss_gamma', (1.0, 2.0, 5.0))
        }
    ))
]

trials = Trials()

best = fmin(
    train,
    space=hyper_space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials
)

print(f'best configuration: {best}')
print(f'best accuracy: {-min(trials.losses())}')
