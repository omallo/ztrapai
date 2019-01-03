from fastai.callbacks import *
from fastai.vision import *
from hyperopt import fmin, tpe, hp, space_eval, Trials

from preact_resnet import *
from resnet import *


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


@dataclass
class MultiTrainMetricTrackerCallback(TrackerCallback):
    def on_train_begin(self, **kwargs):
        old_best = self.best if hasattr(self, 'best') else None
        super().on_train_begin(**kwargs)
        self.best = old_best or self.best

    def on_epoch_end(self, epoch, **kwargs):
        current = self.get_monitor_value()
        if self.operator(current, self.best):
            self.best = current


class ModelConfig:
    def __init__(self, factory, split_func, pretrained):
        self.factory = factory
        self.split_func = split_func
        self.pretrained = pretrained


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        loss = -(1 - pt) ** self.gamma * logpt

        return loss.mean()


def log(*args):
    print(*args, flush=True)


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


def create_learner(data, model_type, models_base_path, dropout, loss_func):
    model_config = get_model_config(model_type)
    return create_cnn(
        data,
        model_config.factory,
        pretrained=model_config.pretrained,
        ps=dropout,
        split_on=model_config.split_func,
        metrics=[accuracy],
        path=models_base_path,
        loss_func=loss_func
    )


def get_model_config(model_type):
    if model_type == 'resnet34':
        return ModelConfig(models.resnet34, resnet_split, True)
    elif model_type == 'resnet50':
        return ModelConfig(models.resnet50, resnet_split, True)
    elif model_type == 'resnet34_small':
        return ModelConfig(lambda pretrained: ResNet34(), None, False)
    elif model_type == 'preact_resnet34':
        return ModelConfig(lambda pretrained: PreActResNet34(), None, False)
    else:
        raise Exception(f'Unsupported model type "{model_type}"')


def get_loss_func(loss_config):
    if loss_config['type'] == 'cce':
        return nn.CrossEntropyLoss()
    elif loss_config['type'] == 'focal':
        return FocalLoss(gamma=loss_config['gamma'])
    else:
        raise Exception(f'Unsupported loss type "{loss_config["type"]}"')


def bootstrap_training(model_type):
    log(f'bootstraping the training for model "{model_type}"\n')

    models_base_path = Path('/artifacts')

    data = create_data(batch_size=64)
    learn = create_learner(data, model_type, models_base_path, 0.2, nn.CrossEntropyLoss())

    model_saving = MultiTrainSaveModelCallback(learn, monitor='accuracy', mode='max', name=model_type)
    early_stopping = MultiTrainEarlyStoppingCallback(learn, monitor='accuracy', mode='max', patience=1, min_delta=1e-3)

    learn.callbacks = [model_saving, early_stopping]

    freeze_lr = 1e-2
    unfreeze_lr = 1e-3

    model_config = get_model_config(model_type)

    if model_config.pretrained:
        log('bootstrap training with freezed model')
        learn.freeze()
        early_stopping.patience = 3
        learn.fit(100, lr=freeze_lr)
        log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')

    log('bootstrap training with unfreezed model and differential learning rates')
    learn.unfreeze()
    early_stopping.patience = 3
    learn.fit(100, lr=slice(unfreeze_lr))
    log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')

    log('bootstrap training with unfreezed model and one-cycle learning rate scheduling')
    cycle_len = 10
    early_stopping.patience = cycle_len - 1
    early_stopping.early_stopped = False
    while not early_stopping.early_stopped:
        learn.fit_one_cycle(cycle_len, max_lr=unfreeze_lr)
        log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')

    return model_saving.best


def train(space):
    model_type = space['model']
    dropout = space['dropout']
    loss_config = space['loss']
    lr_scheduler_config = space['lr_scheduler']
    mixup_config = space['mixup']

    loss_func = get_loss_func(loss_config)

    models_base_path = Path('/artifacts')

    best_bootstraping_score = None
    if not os.path.isfile(f'{models_base_path}/models/{model_type}.pth'):
        best_bootstraping_score = bootstrap_training(model_type)

    log(f'\ntraining with hyper parameters: {space}')

    data = create_data(batch_size=64)
    learn = create_learner(data, model_type, models_base_path, dropout, loss_func)

    if mixup_config['enabled']:
        learn = learn.mixup(alpha=mixup_config['alpha'])

    model_saving = MultiTrainSaveModelCallback(learn, monitor='accuracy', mode='max', name=model_type)
    early_stopping = MultiTrainEarlyStoppingCallback(learn, monitor='accuracy', mode='max', patience=1, min_delta=1e-3)
    best_score_tracker = MultiTrainMetricTrackerCallback(learn, monitor='accuracy', mode='max')

    if best_bootstraping_score:
        model_saving.best = best_bootstraping_score
        early_stopping.best = best_bootstraping_score

    # TODO: check whether previous metrics would also be available on the learner and decide which one to use
    previous_scores = list(filter(lambda l: l is not None, trials.losses()))
    if len(previous_scores) > 0:
        # TODO: should be done per model type
        best_score_to_restore = min(previous_scores) if model_saving.operator == np.less else -min(previous_scores)
        log(f'restoring best {model_saving.monitor}: {best_score_to_restore:.6f}')
        model_saving.best = best_score_to_restore
        early_stopping.best = best_score_to_restore

    learn.callbacks = [model_saving, early_stopping, best_score_tracker]

    learn.load(model_type)

    if lr_scheduler_config['type'] == 'one_cycle':
        lr = 1e-3
        cycle_len = lr_scheduler_config['cycle_len']
        early_stopping.patience = cycle_len - 1
        early_stopping.early_stopped = False
        learn.unfreeze()
        while not early_stopping.early_stopped:
            learn.fit_one_cycle(cycle_len, max_lr=lr)
            log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}')
            log(f'--> best {best_score_tracker.monitor} of current optimization run: {best_score_tracker.best:.6f}')
    else:
        raise Exception(f'Unsupported lr scheduler type "{lr_scheduler_config["type"]}"')

    best_score = best_score_tracker.best
    if isinstance(best_score, Tensor):
        best_score = best_score.item()
    if model_saving.operator != np.less:
        best_score = -best_score

    log(f'loss of current optimization run: {best_score:.6f}\n')

    return best_score


if os.path.isdir('/storage/models/ztrapai/cifar10/models'):
    log('restoring models')
    shutil.copytree('/storage/models/ztrapai/cifar10/models', '/artifacts/models')

hyper_space = {
    'model': hp.choice('model', ('resnet34_small',)),
    'dropout': hp.quniform('dropout', .5, 8.5, 1) / 10,
    'loss': hp.choice('loss', (
        {
            'type': 'cce'
        },
        {
            'type': 'focal',
            'gamma': hp.quniform('focal_loss_gamma', .5, 5.5, 1)
        }
    )),
    'lr_scheduler': hp.choice('lr_scheduler', (
        {
            'type': 'one_cycle',
            'cycle_len': hp.choice('one_cycle_lr_scheduler_cycle_len', (5, 10))
        },
    )),
    'mixup': hp.choice('mixup', (
        {
            'enabled': False
        },
        {
            'enabled': True,
            'alpha': hp.quniform('mixup_alpha', 2.5, 5.5, 1) / 10
        }
    ))
}

trials = Trials()
if False and os.path.isfile('/storage/models/ztrapai/cifar10/trials.p'):
    log('restoring persisted trials')
    shutil.copy('/storage/models/ztrapai/cifar10/trials.p', '/artifacts/trials.p')
    with open('/artifacts/trials.p', 'rb') as trials_file:
        trials = pickle.load(trials_file)

best = fmin(
    train,
    space=hyper_space,
    algo=tpe.suggest,
    max_evals=len(trials.trials) + 10,
    trials=trials
)

with open('/artifacts/trials.p', 'wb') as trials_file:
    pickle.dump(trials, trials_file)

log(f'best hyper parameter configuration: {space_eval(hyper_space, best)}')
log(f'best score: {-min(trials.losses())}')
