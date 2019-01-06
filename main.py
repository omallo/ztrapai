from fastai.callbacks import *
from fastai.vision import *
from hyperopt import fmin, tpe, hp, space_eval, Trials

from preact_resnet import *
from resnet import *


@dataclass
class MultiTrainSaveModelCallback(SaveModelCallback):
    def on_train_begin(self, **kwargs):
        if os.path.isfile(self.get_metric_file_path()):
            with open(self.get_metric_file_path(), 'r') as metric_file:
                self.best = float(metric_file.readline())
        if not hasattr(self, 'best'):
            super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, **kwargs):
        super().on_epoch_end(epoch, **kwargs)
        with open(self.get_metric_file_path(), 'w') as metric_file:
            metric = get_tensor_item(self.best)
            metric_file.write(f'{metric}\n')

    def get_metric_file_path(self):
        return self.learn.path / f'{self.learn.model_dir}/{self.name}_best_{self.monitor}.txt'


@dataclass
class MultiTrainEarlyStoppingCallback(EarlyStoppingCallback):
    def on_train_begin(self, **kwargs):
        old_best = self.best if hasattr(self, 'best') else None
        super().on_train_begin(**kwargs)
        self.best = old_best or self.best

    def on_epoch_end(self, epoch, **kwargs):
        self.early_stopped = super().on_epoch_end(epoch, **kwargs)
        return self.early_stopped


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


def get_tensor_item(value):
    return value.item() if isinstance(value, Tensor) else value


def resnet_split(model):
    return (model[0][6], model[1])


def create_data(batch_size, tfms_enabled):
    return (
        ImageItemList
            .from_folder(untar_data('https://s3.amazonaws.com/fast-ai-imageclas/cifar100'))
            .random_split_by_pct(valid_pct=0.2, seed=42)
            .label_from_folder()
            .transform(get_transforms() if tfms_enabled else None)
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


def space_eval_trial(space, trial):
    vals = trial['misc']['vals']

    # unpack the one-element lists to values
    # and skip over the 0-element lists
    unpacked_vals = {}
    for k, v in list(vals.items()):
        if v:
            unpacked_vals[k] = v[0]

    return space_eval(space, unpacked_vals)


def get_model_config(model_type):
    if model_type == 'resnet34':
        return ModelConfig(models.resnet34, resnet_split, True)
    elif model_type == 'resnet50':
        return ModelConfig(models.resnet50, resnet_split, True)
    elif model_type == 'resnet34_small':
        return ModelConfig(lambda pretrained: ResNet34(), None, False)
    elif model_type == 'preact_resnet18':
        return ModelConfig(lambda pretrained: PreActResNet18(), None, False)
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

    data = create_data(batch_size=64, tfms_enabled=True)
    learn = create_learner(data, model_type, models_base_path, 0.2, nn.CrossEntropyLoss())

    model_saving = MultiTrainSaveModelCallback(learn, monitor='accuracy', mode='max', name=model_type)
    early_stopping = MultiTrainEarlyStoppingCallback(learn, monitor='accuracy', mode='max', patience=1, min_delta=1e-3)

    learn.callbacks = [model_saving, early_stopping]

    freeze_lr = 1e-2
    unfreeze_lr = 1e-3

    model_config = get_model_config(model_type)

    if model_config.pretrained:
        log('bootstrap training with freezed model and fixed learning rate')
        learn.freeze()
        early_stopping.patience = 3
        learn.fit(100, lr=freeze_lr)
        log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')

        log('bootstrap training with unfreezed model and differential learning rates')
        learn.unfreeze()
        early_stopping.patience = 3
        learn.fit(100, lr=slice(unfreeze_lr))
        log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')
    else:
        log('bootstrap training with unfreezed model and fixed learning rate')
        learn.unfreeze()
        early_stopping.patience = 3
        learn.fit(100, lr=unfreeze_lr)
        log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')

    log('bootstrap training with unfreezed model and one-cycle learning rate scheduling')
    cycle_len = 10
    early_stopping.patience = cycle_len - 1
    early_stopping.early_stopped = False
    while not early_stopping.early_stopped:
        learn.fit_one_cycle(cycle_len, max_lr=slice(unfreeze_lr))
        log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}\n')


def train(hyperparams):
    model_type = hyperparams['model']
    dropout = hyperparams['dropout']
    loss_config = hyperparams['loss']
    lr_scheduler_config = hyperparams['lr_scheduler']
    tfms_config = hyperparams['transforms']
    mixup_config = hyperparams['mixup']

    loss_func = get_loss_func(loss_config)

    models_base_path = Path('/artifacts')

    if not os.path.isfile(f'{models_base_path}/models/{model_type}.pth'):
        bootstrap_training(model_type)

    log(f'\nhyper parameters: {hyperparams}')

    learn = create_learner(
        create_data(batch_size=64, tfms_enabled=tfms_config['enabled']),
        model_type,
        models_base_path,
        dropout,
        loss_func
    )

    if mixup_config['enabled']:
        learn = learn.mixup(alpha=mixup_config['alpha'])

    model_saving = MultiTrainSaveModelCallback(learn, monitor='accuracy', mode='max', name=model_type)
    early_stopping = MultiTrainEarlyStoppingCallback(learn, monitor='accuracy', mode='max', patience=1, min_delta=1e-3)

    learn.callbacks = [model_saving, early_stopping]

    learn.load(model_type)

    if lr_scheduler_config['type'] == 'one_cycle':
        lr = 1e-3
        cycle_len = lr_scheduler_config['cycle_len']
        early_stopping.patience = cycle_len - 1
        early_stopping.early_stopped = False
        learn.unfreeze()
        while not early_stopping.early_stopped:
            learn.fit_one_cycle(cycle_len, max_lr=slice(lr))
            log(f'--> best overall {model_saving.monitor}: {model_saving.best:.6f}')
            log(f'--> best {early_stopping.monitor} of current optimization run: {early_stopping.best:.6f}')
    else:
        raise Exception(f'Unsupported lr scheduler type "{lr_scheduler_config["type"]}"')

    best_score = get_tensor_item(early_stopping.best)
    if model_saving.operator != np.less:
        best_score = -best_score

    log(f'loss of current optimization run: {best_score:.6f}\n')

    return best_score


def main():
    if os.path.isdir('/storage/models/ztrapai/cifar10/models'):
        log('restoring models')
        shutil.copytree('/storage/models/ztrapai/cifar10/models', '/artifacts/models')

    hyperspace = {
        'model': hp.choice('model', ('preact_resnet18',)),
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
        'transforms': hp.choice('transforms', (
            {
                'enabled': False
            },
            {
                'enabled': True
            }
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
        space=hyperspace,
        algo=tpe.suggest,
        max_evals=len(trials.trials) + 10,
        trials=trials
    )

    with open('/artifacts/trials.p', 'wb') as trials_file:
        pickle.dump(trials, trials_file)

    best_hyperparams = space_eval(hyperspace, best)

    log(f'best hyper parameters: {best_hyperparams}')
    log(f'best score: {min(trials.losses())}')

    best_model_type = best_hyperparams['model']
    learn = create_learner(
        create_data(batch_size=64, tfms_enabled=best_hyperparams['transforms']['enabled']),
        best_model_type,
        Path('/artifacts'),
        best_hyperparams['dropout'],
        get_loss_func(best_hyperparams['loss']))

    learn.load(best_model_type)

    prediction_logits, true_categories = learn.TTA(ds_type=DatasetType.Valid)
    prediction_accuracy = accuracy(prediction_logits, true_categories)
    log(f'prediction accuracy with TTA: {prediction_accuracy}')


if __name__ == '__main__':
    main()
