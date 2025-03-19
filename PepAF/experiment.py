import copy
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch_geometric
from joblib import Parallel, delayed

from fpa import Single_Prediction, PMHC_Prediction, PDBBind_Prediction
from net import ASMNet, ASMNet_PMHC
from amnet.net import AMNet
from smnet.net import SMNet
from util import Logger, Saver, EarlyStopping, Weighted_MSELoss

class FPAExperiment:
    def __init__(self, task=None, split_method='test', split_frac=[0.8, 0.1, 0.1],
                 prot_gcn_dims=[128, 128, 128], prot_fc_dims=[1024, 128],
                 pep_gcn_dims=[128, 128, 128], pep_fc_dims=[1024, 128],
                 smiles_gcn_dims=[128, 128, 128], smiles_fc_dims=[1024, 128],
                 mlp_dims=[1024, 512], mlp_dropout=0.25, num_pos_emb=16, num_rbf=16,
                 contact_cutoff=8., batch_size=256, onthefly=True, output_dir='../output'):
        
        self.saver = Saver(output_dir)
        self.split_method = split_method
        self.batch_size = batch_size
        self.task = task

        if task == 'pdbbind' or task == 'pmhc':
            self.dataset = self._initialize_dataset(task, onthefly, num_pos_emb, num_rbf, contact_cutoff)

        self._task_data_df_split = None
        self._task_loader = None
        self.devices = self._set_device()

        self.amnet_config = self._build_amnet_config(prot_gcn_dims, prot_fc_dims, pep_gcn_dims, pep_fc_dims, mlp_dims, mlp_dropout)
        self.smnet_config = self._build_smnet_config(prot_gcn_dims, prot_fc_dims, smiles_gcn_dims, smiles_fc_dims, mlp_dims, mlp_dropout)

        self.build_model()
        self.count_parameters()

    def _initialize_dataset(self, task, onthefly, num_pos_emb, num_rbf, contact_cutoff):
        if task == 'pdbbind':
            return PDBBind_Prediction(onthefly=onthefly, num_pos_emb=num_pos_emb, num_rbf=num_rbf, contact_cutoff=contact_cutoff)
        elif task == 'single':
            return Single_Prediction(num_pos_emb=num_pos_emb, num_rbf=num_rbf, contact_cutoff=contact_cutoff)
        elif task == 'pmhc':
            return PMHC_Prediction(onthefly=onthefly, num_pos_emb=num_pos_emb, num_rbf=num_rbf, contact_cutoff=contact_cutoff)
        else:
            raise ValueError(f'Unknown task: {task}')

    def _set_device(self):
        n_gpus = torch.cuda.device_count()
        device_index = 1 % n_gpus
        return [torch.device(f'cuda:{device_index}') if n_gpus > 1 else torch.device('cuda:0')]

    def _build_amnet_config(self, prot_gcn_dims, prot_fc_dims, pep_gcn_dims, pep_fc_dims, mlp_dims, mlp_dropout):
        return {
            'prot_emb_dim': 1280,
            'prot_gcn_dims': prot_gcn_dims,
            'prot_fc_dims': prot_fc_dims,
            'pep_gcn_dims': pep_gcn_dims,
            'pep_fc_dims': pep_fc_dims,
            'mlp_dims': mlp_dims,
            'mlp_dropout': mlp_dropout
        }

    def _build_smnet_config(self, prot_gcn_dims, prot_fc_dims, smiles_gcn_dims, smiles_fc_dims, mlp_dims, mlp_dropout):
        return {
            'prot_emb_dim': 1280,
            'prot_gcn_dims': prot_gcn_dims,
            'prot_fc_dims': prot_fc_dims,
            'drug_gcn_dims': smiles_gcn_dims,
            'drug_fc_dims': smiles_fc_dims,
            'mlp_dims': mlp_dims,
            'mlp_dropout': mlp_dropout
        }

    def build_model(self):
        self.amnets = [AMNet(**self.amnet_config).to(self.devices[0]) for _ in range(5)]
        self.smnets = [SMNet(**self.smnet_config).to(self.devices[0]) for _ in range(5)]
        
        if self.task == 'pmhc':
            self.models = [ASMNet_PMHC(amnet, smnet).to(self.devices[0]) for amnet, smnet in zip(self.amnets, self.smnets)]
        else:
            self.models = [ASMNet(amnet, smnet).to(self.devices[0]) for amnet, smnet in zip(self.amnets, self.smnets)]

    def count_parameters(self):
        total_params = sum(param.numel() for param in self.models[0].parameters())
        trainable_params = sum(param.numel() for param in self.models[0].parameters() if param.requires_grad)

    def _get_data_loader(self, dataset, shuffle=False):
        return torch_geometric.loader.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=False, num_workers=0)

    @property
    def task_data_df_split(self):
        if self._task_data_df_split is None:
            self._task_data_df_split = self.dataset.get_split(return_df=True)
        return self._task_data_df_split

    @property
    def task_data(self):
        return self.task_data_df_split[0]

    @property
    def task_df(self):
        return self.task_data_df_split[1]

    @property
    def task_loader(self):
        if self._task_loader is None:
            self._task_loader = {s: self._get_data_loader(self.task_data[s], shuffle=(s == 'train')) for s in self.task_data}
        return self._task_loader

    def _format_predict_df(self, results, test_df=None, pred_type='three'):
        df = self.task_df['test'].copy() if test_df is None else test_df.copy()
        df['y_pred'] = results[f'y_pred_{pred_type}']
        return df


    def predict_pdbbind(self, checkpoints, save_prediction=False, save_df_name='prediction_pdbbind.tsv'):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(checkpoints[i], map_location=self.devices[0]), strict=False)

        test_loader = self.task_loader['test']
        
        rets_list = [
            _test(kwargs={'midx': i + 1, 'model': model, 'test_loader': test_loader, 'device': self.devices[0]})
            for i, model in enumerate(self.models)
        ]

        y_pred_three = np.mean([rets['y_pred_three'] for rets in rets_list], axis=0)
        y_pred_two = np.mean([rets['y_pred_two'] for rets in rets_list], axis=0)
        
        results = {
            'y_pred_three': y_pred_three,
            'y_pred_two': y_pred_two,
            'df': self._format_predict_df({'y_pred_three': y_pred_three, 'y_pred_two': y_pred_two}, pred_type='three')
        }
        
        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')
            print(f"Results have been saved in {save_df_name}")

        return results

    def predict_pmhc(self, checkpoints, save_prediction=False, save_df_name='prediction_pmhc.tsv'):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(checkpoints[i], map_location=self.devices[0]), strict=False)

        test_loader = self.task_loader['test']
        
        rets_list = [
            _test(kwargs={'midx': i + 1, 'model': model, 'test_loader': test_loader, 'device': self.devices[0]})
            for i, model in enumerate(self.models)
        ]

        y_pred_three = np.mean([rets['y_pred_three'] for rets in rets_list], axis=0)
        y_pred_two = np.mean([rets['y_pred_two'] for rets in rets_list], axis=0)
        
        results = {
            'y_pred_three': y_pred_three,
            'y_pred_two': y_pred_two,
            'df': self._format_predict_df({'y_pred_three': y_pred_three, 'y_pred_two': y_pred_two}, pred_type='three')
        }
        
        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')
            print(f"Results have been saved in {save_df_name}")

        return results
    
    def predict_single(self, checkpoints, protein_id='7lll_R', peptide_seq='AELAELAEL'):
        print(f"Starting prediction for protein: {protein_id}, peptide: {peptide_seq}")
        dataset = Single_Prediction(protein_id=protein_id, peptide_seq=peptide_seq)
        print("Dataset created.")

        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(checkpoints[i], self.devices[0]), strict=False)
            model.eval()
        print(f"Models loaded successfully.")

        prot_data, pep_data, smi_data = dataset.get_data()

        with torch.no_grad():
            y_mean = sum(
                model(prot_data.to(self.devices[0]), pep_data.to(self.devices[0]), smi_data.to(self.devices[0]))[0].cpu().item()
                for model in self.models
            ) / len(self.models)

        print(f"Predicted binding affinity between {protein_id} and {peptide_seq}: {y_mean}")
        return y_mean

def _test(kwargs):
    midx = kwargs['midx']
    model = kwargs['model']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    model.eval()
    yp_three, yp_two = torch.Tensor(), torch.Tensor()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            xpep = batch['peptide'].to(device)
            xpro = batch['protein'].to(device)
            xdru = batch['smiles'].to(device)
            y_three, y_two = model(xpro, xpep, xdru)
            yp_three = torch.cat([yp_three, y_three.detach().cpu()], dim=0)
            yp_two = torch.cat([yp_two, y_two.detach().cpu()], dim=0)

    return {
        'midx': midx,
        'y_pred_three': yp_three.view(-1).numpy(),
        'y_pred_two': yp_two.view(-1).numpy(),
    }