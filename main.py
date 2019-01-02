from fastai.callbacks import *
from fastai.vision import *
from hyperopt import fmin, tpe, hp, Trials

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


def create_learner(data, model_factory_func, model_split_func, models_base_path, dropout, loss_func):
    return create_cnn(
        data,
        model_factory_func,
        pretrained=True,
        ps=dropout,
        split_on=model_split_func,
        metrics=[accuracy],
        path=models_base_path,
        loss_func=loss_func
    )


def get_model_factory(model_name):
    if model_name == 'resnet34':
        return models.resnet34
    elif model_name == 'resnet50':
        return models.resnet50
    elif model_name == 'resnet34_small':
        return lambda pretrained: ResNet34()
    elif model_name == 'preact_resnet34':
        return lambda pretrained: PreActResNet34()
    else:
        raise Exception(f'Unsupported model type "{model_name}"')


def get_loss_func(loss_config):
    if loss_config['type'] == 'cce':
        return nn.CrossEntropyLoss()
    elif loss_config['type'] == 'focal':
        return FocalLoss(gamma=loss_config['gamma'])
    else:
        raise Exception(f'Unsupported loss type "{loss_config["type"]}"')


def bootstrap_training(model_name, model_factory):
    log(f'bootstraping the training for model "{model_name}"\n')

    models_base_path = Path('/artifacts')

    data = create_data(batch_size=64)
    learn = create_learner(data, model_factory, resnet_split, models_base_path, 0.2, nn.CrossEntropyLoss())

    model_saving = MultiTrainSaveModelCallback(learn, monitor='accuracy', mode='max', name=model_name)
    early_stopping = MultiTrainEarlyStoppingCallback(learn, monitor='accuracy', mode='max', patience=1, min_delta=1e-3)

    learn.callbacks = [model_saving, early_stopping]

    freeze_lr = 1e-2
    unfreeze_lr = 1e-3

    learn.freeze()
    early_stopping.patience = 3
    learn.fit(100, lr=freeze_lr)
    log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')

    learn.unfreeze()
    early_stopping.patience = 3
    learn.fit(100, lr=slice(unfreeze_lr))
    log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')

    cycle_len = 10
    early_stopping.patience = cycle_len - 1
    early_stopping.early_stopped = False
    while not early_stopping.early_stopped:
        learn.fit_one_cycle(cycle_len, max_lr=unfreeze_lr)
        log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')

    return model_saving.best


def train(space):
    model_name = space['model']
    dropout = space['dropout']
    loss_config = space['loss']
    lr_scheduler_config = space['lr_scheduler']

    model_factory = get_model_factory(model_name)
    loss_func = get_loss_func(loss_config)

    models_base_path = Path('/artifacts')

    best_bootstraping_score = None
    if not os.path.isfile(f'{models_base_path}/models/{model_name}.pth'):
        best_bootstraping_score = bootstrap_training(model_name, model_factory)

    log(f'\ntraining with hyper parameters: {space}\n')

    data = create_data(batch_size=64)
    learn = create_learner(data, model_factory, resnet_split, models_base_path, dropout, loss_func)

    model_saving = MultiTrainSaveModelCallback(learn, monitor='accuracy', mode='max', name=model_name)
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
        log(f'restoring best {model_saving.monitor}: {best_score_to_restore:.6f}\n')
        model_saving.best = best_score_to_restore
        early_stopping.best = best_score_to_restore

    learn.callbacks = [model_saving, early_stopping, best_score_tracker]

    learn.load(model_name)

    if lr_scheduler_config['type'] == 'one_cycle':
        lr = 1e-3
        cycle_len = lr_scheduler_config['cycle_len']
        early_stopping.patience = cycle_len - 1
        early_stopping.early_stopped = False
        learn.unfreeze()
        while not early_stopping.early_stopped:
            learn.fit_one_cycle(cycle_len, max_lr=lr)
            log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}')
            log(f'--> best {best_score_tracker.monitor} of current optimization run: {best_score_tracker.best:.6f}\n')
    else:
        raise Exception(f'Unsupported lr scheduler type "{lr_scheduler_config["type"]}"')

    best_score = best_score_tracker.best
    if isinstance(best_score, Tensor):
        best_score = best_score.item()
    if model_saving.operator != np.less:
        best_score = -best_score

    log(f'\nloss of current optimization run: {best_score:.6f}')

    return best_score


if os.path.isdir('/storage/models/ztrapai/cifar10/models'):
    log('restoring models')
    shutil.copytree('/storage/models/ztrapai/cifar10/models', '/artifacts/models')

hyper_space = {
    'model': hp.choice('model', ('resnet34', 'resnet34_small', 'preact_resnet34')),
    'dropout': hp.choice('dropout', (0.1, 0.2, 0.5, 0.8)),
    'loss': hp.choice('loss', (
        {
            'type': 'cce'
        },
        {
            'type': 'focal',
            'gamma': hp.choice('focal_loss_gamma', (1.0, 2.0, 5.0))
        }
    )),
    'lr_scheduler': hp.choice('lr_scheduler', (
        {
            'type': 'one_cycle',
            'cycle_len': hp.choice('one_cycle_lr_scheduler_cycle_len', (5, 10))
        },
    ))
}

trials = Trials()
if os.path.isfile('/storage/models/ztrapai/cifar10/trials.p'):
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

# TODO: check out space_eval() from hyperopt

log(f'best hyper parameter configuration: {best}')
log(f'best score: {-min(trials.losses())}')
