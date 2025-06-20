from utils import *

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
torch.autograd.set_detect_anomaly(True)

SCHEDULER_FIRSTSTEP = [None]
SCHEDULER_BATCHWISE = [None]
SCHEDULER_EPOCHWISE = [None]


class Trainer(object):
    def __init__(self, conf, wandb_run, logging):
        self.logging = logging

        self.device     = conf.experiment.device_type
        self.wandb_run  = wandb_run
        self.pred_model = 'ArcFDI'

        self.optimizer_name = conf.train_params.optimizer_name
        self.scheduler_name = conf.train_params.scheduler_name

        self.batch_size    = conf.train_params.batch_size
        self.num_epochs    = conf.train_params.num_epochs
        self.learning_rate = conf.train_params.learning_rate
        self.weight_decay  = conf.train_params.weight_decay

        self.num_patience  = conf.train_params.early_stopping.num_patience
        self.best_criteria = conf.train_params.early_stopping.best_criteria

        self.dfi_loss_coef = conf.train_params.loss_objective.main_coef
        self.arc_loss_coef = conf.train_params.loss_objective.aux_coef

        self.checkpoint_path = f'{conf.path.checkpoint}/{conf.wandb.project_name}_{conf.wandb.session_name}_fold_{conf.experiment.fold_num}/'
        if not conf.experiment.testing_mode:
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self.save_name = conf.wandb.session_name
        self.lookup_values = {'dfi/y': [], 'dfi/yhat': [], 'arc_drug/y': [], 'arc_drug/yhat': [], 'arc_food/y': [], 'arc_food/yhat': []}

        self.best_valid_metric = {
            'dfi/auroc': 0.,      'dfi/auprc': 0.,      'dfi/f1score': 0.,      'dfi/accuracy': 0.,
            'arc_drug/auroc': 0., 'arc_drug/auprc': 0., 'arc_drug/f1score': 0., 'arc_drug/accuracy': 0.,
            'arc_food/auroc': 0., 'arc_food/auprc': 0., 'arc_food/f1score': 0., 'arc_food/accuracy': 0.}
        self.current_valid_metric = {
            'dfi/auroc': 0.,      'dfi/auprc': 0.,      'dfi/f1score': 0.,      'dfi/accuracy': 0.,
            'arc_drug/auroc': 0., 'arc_drug/auprc': 0., 'arc_drug/f1score': 0., 'arc_drug/accuracy': 0.,
            'arc_food/auroc': 0., 'arc_food/auprc': 0., 'arc_food/f1score': 0., 'arc_food/accuracy': 0.}
        assert self.best_criteria in self.best_valid_metric.keys()

        pickle.dump(conf, open(self.checkpoint_path + f'config.pkl', 'wb'))

        self.rank = 0

    def print0(self, text: str):
        if self.rank == 0:
            self.logging.info(text)

    def calculate_losses(self, output_batch):
        total_loss = []
        dfi_loss   = nn.BCELoss()
        arc_loss   = Masked_NLLLoss([1., 1.], self.device)

        total_loss.append(self.dfi_loss_coef * dfi_loss(output_batch['dfi/yhat'],output_batch['dfi/y'])) 
        total_loss.append(self.arc_loss_coef * arc_loss(output_batch['arc_drug/yhat'],output_batch['arc_drug/y']))
        total_loss.append(self.arc_loss_coef * arc_loss(output_batch['arc_food/yhat'],output_batch['arc_food/y']))
        total_loss = [t for t in total_loss if isinstance(t, torch.Tensor)]

        return total_loss


    def check_valid_progress(self):

        return self.best_valid_metric[self.best_criteria] < self.current_valid_metric[self.best_criteria]

    def reset_lookup_values(self):
        self.lookup_values = {'dfi/y': [], 'dfi/yhat': [], 'arc_drug/y': [], 'arc_drug/yhat': [], 'arc_food/y': [], 'arc_food/yhat': []}

        return

    def store_lookup_values(self, output_batch):
        for k in self.lookup_values.keys():
            if isinstance(output_batch[k], torch.Tensor):
                numpified = numpify(output_batch[k])
                # if 'arc' in k:
                #     numpified_masks = numpify(output_batch[k.split('/')[0]+'/m']).astype(bool)
                #     numpified = numpified[numpified_masks]
                self.lookup_values[k].append(numpified)

        return

    def wandb_lookup_values(self, label, epoch, losses):
        wandb_dict = {f'{label}/step': epoch}

        self.print0(f"Loss Report for Epoch #{epoch}")
        self.print0(f"Batchwise Loss for {label} Data Partition")
        for idx, loss in enumerate(losses):
            self.print0(f"Batchwise Loss Term Index {idx+1}: {loss:.3f}")
            wandb_dict[f'{label}/loss/idx{idx+1}'] = loss

        if len(self.lookup_values['dfi/yhat']) > 0:
            y    = np.concatenate(self.lookup_values['dfi/y']).astype(int)
            yhat = np.concatenate(self.lookup_values['dfi/yhat'])

            try:
                wandb_dict[f'{label}/dfi/auroc']    = auc(y, yhat)
                wandb_dict[f'{label}/dfi/auprc']    = aup(y, yhat)
                yhat                                = (yhat > 0.5).reshape(-1)
                wandb_dict[f'{label}/dfi/f1score']  = f1(y, yhat)
                wandb_dict[f'{label}/dfi/accuracy'] = acc(y, yhat)
            except:
                pass

        if len(self.lookup_values['arc_drug/yhat']) > 0:
            y    = np.concatenate(self.lookup_values['arc_drug/y']).astype(int)
            yhat = np.concatenate(self.lookup_values['arc_drug/yhat'])
            yhat = yhat[:,1]

            try:
                wandb_dict[f'{label}/arc_drug/auroc']    = auc(y, yhat)
                wandb_dict[f'{label}/arc_drug/auprc']    = aup(y, yhat)
                yhat                                     = (yhat > 0.5).reshape(-1)
                wandb_dict[f'{label}/arc_drug/f1score']  = f1(y, yhat)
                wandb_dict[f'{label}/arc_drug/accuracy'] = acc(y, yhat)
            except:
                pass

        if len(self.lookup_values['arc_food/yhat']) > 0:
            try:
                y    = np.concatenate(self.lookup_values['arc_food/y']).astype(int)
                yhat = np.concatenate(self.lookup_values['arc_food/yhat'])
                yhat = yhat[:,1]

                wandb_dict[f'{label}/arc_food/auroc']    = auc(y, yhat)
                wandb_dict[f'{label}/arc_food/auprc']    = aup(y, yhat)
                yhat                                     = (yhat > 0.5).reshape(-1)
                wandb_dict[f'{label}/arc_food/f1score']  = f1(y, yhat)
                wandb_dict[f'{label}/arc_food/accuracy'] = acc(y, yhat)
            except:
                pass

        if label == 'valid':
            for k, v in wandb_dict.items():
                temp = k.removeprefix(label+'/')
                self.current_valid_metric[temp] = v

        if self.rank == 0:
            self.wandb_run.log(wandb_dict)

        return wandb_dict

    def check_early_stopping(self):
        if not self.check_valid_progress():
            self.print0("Validation Didn't Improve....")
            self.num_patience -= 1
            if self.num_patience == 0: 
                self.print0("Early Stopping...............")
                return True

        return False

    def save_checkpoint_per_epoch(self, model):
        for metric in ['auroc', 'auprc', 'f1score', 'accuracy']:
            if self.best_valid_metric[f'dfi/{metric}'] < self.current_valid_metric[f'dfi/{metric}']:
                self.print0(f"Saving Model Checkpoint with Best Validation Performance... [{metric}]")
                torch.save(model.module.state_dict(), self.checkpoint_path + f'best_epoch_{metric}.mdl')
                self.best_valid_metric[f'dfi/{metric}'] = self.current_valid_metric[f'dfi/{metric}']

        return

    def get_optimizer(self, model):
        if self.optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), 
                                   lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), 
                                    lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise
        return optimizer

    def get_scheduler(self):

        return DummyScheduler()


    def train_step(self, model, data, epoch=0):
        model.train()
        batchwise_loss = []

        if self.pred_model in SCHEDULER_FIRSTSTEP:
            self.scheduler.step()

        for idx, batch in enumerate(data):
            self.optimizer.zero_grad()
            batch = model(batch)
            loss = self.calculate_losses(batch)
            sum(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            if self.pred_model in SCHEDULER_BATCHWISE: 
                self.scheduler.step()
            sleep(0.01)
            batchwise_loss.append(list(map(lambda x: x.item(), loss)))
            self.store_lookup_values(batch)
        if self.pred_model in SCHEDULER_EPOCHWISE:
            self.scheduler.step()

        return np.array(batchwise_loss).mean(0).tolist(), model    

    @torch.no_grad()
    def eval_step(self, model, data):
        model.eval()
        batchwise_loss = []

        for idx, batch in enumerate(data):
            batch = model(batch)            
            loss = self.calculate_losses(batch)
            sleep(0.01)
            batchwise_loss.append(list(map(lambda x: x.item(), loss)))
            self.store_lookup_values(batch)

        return np.array(batchwise_loss).mean(0).tolist(), model

    def train_valid(self, model, train, valid=None, train_sampler=None):
        self.train_steps = len(train)
        num_ranks = torch.cuda.device_count()
        self.print0(f"RANK: 0 | Training Batches: {len(train)}, Validation Batches: {len(valid)}")
        print("Training Model on RANK: ", self.rank)
        EARLY_STOPPING = False

        # model = model.to(self.rank)
        model.apply(initialize)
        self.optimizer = self.get_optimizer(model)
        self.scheduler = self.get_scheduler()

        for epoch in range(self.num_epochs):
            if train_sampler: train_sampler.set_epoch(epoch)
            train_loss, model = self.train_step(model, train, epoch)
            self.wandb_lookup_values('train', epoch, train_loss) 
            self.reset_lookup_values()

            eval_loss, _ = self.eval_step(model, valid)
            self.wandb_lookup_values('valid', epoch, eval_loss)
            self.reset_lookup_values()

            if self.check_early_stopping(): 
                break

            if self.rank == 0:
                self.save_checkpoint_per_epoch(model)

            self.print0("")

        return model 

    @torch.no_grad()
    def test(self, model, test):
        self.print0(f"RANK: 0 | Test Batches: {len(test)}")
        print("Testing Model on RANK: ", self.rank)

        eval_loss, _ = self.eval_step(model, test)
        self.wandb_lookup_values('test',0, eval_loss)
        self.reset_lookup_values()

        return model


if __name__ == '__main__':
    import wandb
    import json
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from omegaconf import OmegaConf
    from dataloader.ArcFDI import DatasetPreload, collate_fn
    from model.ArcFDI import Net
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    import os
    import random

    num_cpus = os.cpu_count()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    torch.set_num_threads(1)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['OPENBLAS_NUM_THREADS'] = "1"

    def setup_seed(seed):
        print("Setting the Random Seed to ", seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def setup_wandb(conf):
        print("Setting the Wandb Run Object...")
        wandb_init = dict()
        wandb_init['project'] = conf.wandb.project_name
        wandb_init['group'] = conf.wandb.session_name
        if not conf.experiment.testing_mode:
            wandb_init['name'] = f'training_{conf.dataprep.dataset}_{conf.dataprep.version}_{conf.model_params.model_type}_{conf.model_params.compound_features}' 
        else:
            wandb_init['name'] = f'testing_{conf.dataprep.dataset}_{conf.dataprep.version}_{conf.model_params.model_type}_{conf.model_params.compound_features}'
        wandb_init['notes'] = conf.wandb.session_name
        os.environ['WANDB_START_METHOD'] = 'thread'

        return wandb_init

    def reset_wandb_env():
        exclude = {'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_API_KEY',}
        for k, v in os.environ.items():
            if k.startswith('WANDB_') and k not in exclude:
                del os.environ[k]

    conf = OmegaConf.load(f'./settings.yaml')['dev']    
    print(conf)

    setup_seed(conf.experiment.random_seed)

    dataset    = DatasetPreload(conf)
    tr, va, te = dataset.make_preset_splits()

    trainloader = DataLoader(dataset, batch_size=128, sampler=SubsetRandomSampler(tr), collate_fn=collate_fn)
    validloader = DataLoader(dataset, batch_size=128, sampler=SubsetRandomSampler(va), collate_fn=collate_fn)
    testloader  = DataLoader(dataset, batch_size=128, sampler=SubsetRandomSampler(te), collate_fn=collate_fn)

    model   = Net(conf)
    model   = model.to(conf.experiment.device_type)

    wandb_init = setup_wandb(conf)
    wandb_run = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'))
    wandb_run.define_metric('train/step'); wandb_run.define_metric('train/*', step_metric='train/step')
    wandb_run.define_metric('valid/step'); wandb_run.define_metric('valid/*', step_metric='valid/step')
    wandb_run.define_metric('test/step');  wandb_run.define_metric('test/*', step_metric='test/step')
    wandb_run.watch(model, log="gradients", log_freq=10)

    trainer = Trainer(conf, wandb_run)
    model = trainer.train_valid(model, trainloader, validloader)
    trainer.wandb_run.finish()

    print(model)