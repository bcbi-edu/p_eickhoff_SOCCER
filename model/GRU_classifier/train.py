import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from dataloader import GameFeatures
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint
import os 
from argparse import ArgumentParser
import pandas as pd 
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GRUClassifier(pl.LightningModule):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The GRU takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru = nn.GRU(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.dense = nn.Linear(hidden_dim, hidden_dim//2)
        self.hidden2tag = nn.Linear(hidden_dim//2, tagset_size)
        self.criterion = nn.NLLLoss()

    def forward(self, embs):
        """
        Args:
            embs (Tensor)): (Lx1xH1)

        Returns:
            tag_scores [Tensor]: (BxL) The alpha value for attention 
        """

        gru_out, _ = self.gru(embs)

        hidden = self.dense(gru_out.view(embs.size(0), -1))
        tag_space = self.hidden2tag(hidden)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def training_step(self, batch, batch_nb):
        emb, label = batch
        out = self(emb.permute(1,0,2))
        loss = self.criterion(out, label.squeeze(0))
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.002)

    def test_step(self, batch, batch_nb):
        emb, label = batch
        out = self(emb.permute(1,0,2))
        return {'val_acc': (out.argmax(dim=1) == label).float().mean()}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_acc': avg_loss}
        return {'val_acc': avg_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_acc': avg_loss}
        return {'val_acc': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        emb, label = batch
        out = self(emb.permute(1,0,2))

        return {'val_acc': (out.argmax(dim=1) == label).float().mean()}


def main(hparams):
    
    use_labels = hparams.use_labels.split(',') #['goal_home']
    use_labels = None if use_labels == [''] else use_labels
    label_len = 2**len(use_labels) if use_labels is not None else 1024
    
    if hparams.run_inference:
        game_val = GameFeatures(base_dir=hparams.dataset_path, subset='val', use_labels=use_labels)
        val_loader = DataLoader(game_val, batch_size=1)

        game_test = GameFeatures(base_dir=hparams.dataset_path, subset='test', use_labels=use_labels)
        test_loader = DataLoader(game_test, batch_size=1)

        dataloaders = {
            'val':val_loader,
            'test':test_loader,
        }

        model = GRUClassifier(768, 512, label_len)
        model.load_state_dict(torch.load(hparams.model_save_path))
        model.to(DEVICE)
        
        for subset in ['val', 'test']:
            print(f"Working on {subset} set")

            dataloader = dataloaders[subset]
            label_res = []
            pred_res  = []

            for batch in tqdm(dataloader):

                emb, label = batch
                out = emb.permute(1,0,2)
                out = model(out.to(DEVICE))
                label_res.extend(label.reshape(label.size(1), -1).tolist())
                pred_res.extend(torch.exp(out).tolist())
            
            if not os.path.exists(hparams.result_save_path):
                os.makedirs(hparams.result_save_path)
                print(f"Creating model saving folder: {hparams.result_save_path}")
            
            
            pd.DataFrame(label_res).to_csv(f'{hparams.result_save_path}/{subset}-label.csv')
            pd.DataFrame(pred_res).to_csv(f'{hparams.result_save_path}/{subset}-pred.csv')

    else:
            
        # train!
        game_train = GameFeatures(base_dir=hparams.dataset_path, subset='train', use_labels=use_labels)
        train_loader = DataLoader(game_train, batch_size=1)

        game_val = GameFeatures(base_dir=hparams.dataset_path, subset='val', use_labels=use_labels)
        val_loader = DataLoader(game_val, batch_size=1)
        
        model = GRUClassifier(768, 512, label_len)

        # default used by the Trainer
        checkpoint_callback = ModelCheckpoint(
            filepath=hparams.log_save_path,
            save_top_k=True,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix='alpha',
            period=1
        )

        trainer = pl.Trainer(gpus=1, max_epochs=5,
                            default_root_dir=hparams.log_save_path,
                            checkpoint_callback=checkpoint_callback)    
        
        trainer.fit(model, train_loader, val_dataloaders=val_loader) 
        
        model_save_folder = os.path.dirname(hparams.model_save_path)
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)
            print(f"Creating model saving folder: {model_save_folder}")
            
        torch.save(model.state_dict(), hparams.model_save_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run_inference', action='store_true')
    parser.add_argument('--use_labels', default='')
    parser.add_argument('--dataset_path', default='features/')
    parser.add_argument('--log_save_path', default=os.getcwd())
    parser.add_argument('--model_save_path', default='./models/multi-class/all.pth')
    parser.add_argument('--result_save_path', default='./results/all/')
    args = parser.parse_args()

    main(args)

