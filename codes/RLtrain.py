import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
import multiprocessing

# rdkit
from rdkit import Chem, DataStructs

from model import GPT, GPTConfig
from vocabulary import read_vocabulary
from utils import set_seed, sample_SMILES, likelihood, to_tensor, calc_fingerprints
from scoring_function import get_scores, int_div

class RL_trainer():

    def __init__(self, logger, configs):
        self.run_name = configs.run_name
        self.writer = logger
        self.device = configs.device
        self.task = configs.task
        self.target = configs.target
        self.prior_path = configs.prior_path
        self.voc = read_vocabulary(configs.vocab_path)
        self.batch_size = configs.batch_size
        self.n_steps = configs.n_steps
        self.learning_rate = configs.learning_rate
        self.sigma = configs.sigma
        # AC augmentation
        self.AC_aug = configs.AC_aug
        self.memory1 = pd.DataFrame(columns=["smiles", "scores", "seqs", "fps"])
        self.memory2 = pd.DataFrame(columns=["smiles", "scores", "seqs", "fps"])
        self.alpha1 = configs.alpha1
        self.alpha2 = configs.alpha2
        self.t = configs.t
        self.s1 = configs.s1
        self.s2 = configs.s2
        

    def _memory_update(self, smiles, scores, seqs):
        scores = list(scores)
        seqs_list = [seqs[i, :].cpu().numpy() for i in range(len(smiles))]

        mean_coef = 0
        for i in range(len(smiles)):
            if scores[i] < 0:
                continue
            # canonicalized SMILES and fingerprints
            fp, smiles_i = calc_fingerprints([smiles[i]])
            new_data = pd.DataFrame({"smiles": smiles_i[0], "scores": scores[i], "seqs": [seqs_list[i]], "fps": fp[0]})
            self.memory1 = pd.concat([self.memory1, new_data], ignore_index=True, sort=False)

            if self.AC_aug == False:
                continue

            for j in range(len(self.memory1)):
                diff_abs = np.abs(self.memory1["scores"][j] - scores[i])
                if diff_abs > self.alpha1:
                    dist = 1 - DataStructs.FingerprintSimilarity(self.memory1["fps"][j], fp[0])
                    ACI = diff_abs / dist
                    if ACI > self.alpha2 and self.memory1["scores"][j] > scores[i]:
                        self.memory2 = pd.concat([self.memory2, new_data], ignore_index=True, sort=False)
                        break

                    elif ACI > self.alpha2 and self.memory1["scores"][j] < scores[i]:
                        new_data = self.memory1.loc[j]
                        self.memory2.loc[len(self.memory2)] = new_data

        self.memory1 = self.memory1.drop_duplicates(subset=["smiles"])
        self.memory1 = self.memory1.sort_values('scores', ascending=False)
        self.memory1 = self.memory1.reset_index(drop=True)
        self.memory2 = self.memory2.drop_duplicates(subset=["smiles"])
        self.memory2 = self.memory2.reset_index(drop=True)

        # experience replay
        s1 = min(len(self.memory1), self.s1)
        if s1 > 0:
            experience1 = self.memory1.head(min(len(self.memory1), self.t)).sample(s1)
            experience1 = experience1.reset_index(drop=True)

            smiles += list(experience1["smiles"])
            scores += list(experience1["scores"])
            for index in experience1.index:
                seqs = torch.cat((seqs, torch.tensor(experience1.loc[index, "seqs"], dtype=torch.long).view(1, -1).cuda()), dim=0)

        s2 = min(len(self.memory2), self.s2)
        if s2 > 0:
            experience2 = self.memory2.sample(s2)
            experience2 = experience2.reset_index(drop=True)
            smiles += list(experience2["smiles"])
            scores += list(experience2["scores"])
            for j in range(len(experience2)):
                seqs = torch.cat((seqs, torch.tensor(experience2["seqs"][j], dtype=torch.long).view(1, -1).cuda()), dim=0)


        return smiles, np.array(scores), seqs


    def train(self):
        # Initilization
        if not os.path.exists(f'outputs_new/{self.task}_{self.target}'):
            os.makedirs(f'outputs_new/{self.task}_{self.target}')

        prior_config = GPTConfig(self.voc.__len__(), n_layer=8, n_head=8, n_embd=256, block_size=128)
        prior = GPT(prior_config).to(self.device)
        agent = GPT(prior_config).to(self.device)
        optimizer = agent.configure_optimizers(weight_decay=0.1, 
                                                    learning_rate=self.learning_rate, 
                                                    betas=(0.9, 0.95))
        
        prior.load_state_dict(torch.load(self.prior_path), strict=True)
        for param in prior.parameters():
            param.requires_grad = False
        prior.eval()
        agent.load_state_dict(torch.load(self.prior_path), strict=True)
        agent.eval()

        # RL iterations
        for step in tqdm(range(self.n_steps)):
            samples, seqs, _ = sample_SMILES(agent, self.voc, n_mols=self.batch_size)

            scores = get_scores(samples, mode=self.task, target=self.target)
            self.writer.add_scalar('mean score', np.mean(scores), step)
            samples, scores, seqs = self._memory_update(samples, scores, seqs)
        
            prior_likelihood = likelihood(prior, seqs)
            agent_likelihood = likelihood(agent, seqs)
            loss = torch.pow(self.sigma * to_tensor(np.array(scores)) - (prior_likelihood - agent_likelihood), 2)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tensorboard
            self.writer.add_scalar('training loss', loss.item(), step)
            self.writer.add_scalar('loss diff', torch.mean(prior_likelihood - agent_likelihood).item(), step)

            # self.writer.add_scalar('mean score in memory', np.mean(np.array(self.memory1["scores"])), step)
            self.writer.add_scalar('top-1', self.memory1["scores"][0], step)
            self.writer.add_scalar('top-10', np.mean(np.array(self.memory1["scores"][:10])), step)
            self.writer.add_scalar('top-100', np.mean(np.array(self.memory1["scores"][:100])), step)

            self.writer.add_scalar('num_ACs', len(self.memory2), step)


            if (step + 1) % 20 == 0:
                self.writer.add_scalar('top-100-div', int_div(list(self.memory1["smiles"][:100])), step)
                # self.writer.add_scalar('memory-div', int_div(list(self.memory1["smiles"])), step)
            if (step + 1) % 50 == 0:
                self.memory1.to_csv(f'outputs_new/{self.task}_{self.target}/{self.run_name}_step{step + 1}_M1.csv')
                self.memory2.to_csv(f'outputs_new/{self.task}_{self.target}/{self.run_name}_step{step + 1}_M2.csv')

        self.memory1.to_csv(f'outputs_new/{self.task}_{self.target}/{self.run_name}_M1.csv')
        self.memory1.to_csv(f'outputs_new/{self.task}_{self.target}/{self.run_name}_M2.csv')

        print(f'top-1 score: {self.memory1["scores"][0]}')
        print(f'top-10 score: {np.mean(np.array(self.memory1["scores"][:10]))}')
        print(f'top-100 score: {np.mean(np.array(self.memory1["scores"][:100]))}, diversity: {int_div(list(self.memory1["smiles"][:100]))}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--task', type=str, default="docking_mpo")
    parser.add_argument('--target', type=str, default="5HT2B")
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-5)

    parser.add_argument('--AC_aug', type=bool, default=True)
    parser.add_argument('--alpha1', type=float, default=0.5)
    parser.add_argument('--alpha2', type=float, default=2)
    parser.add_argument('--t', type=int, default=100)
    parser.add_argument('--s1', type=int, default=20)
    parser.add_argument('--s2', type=int, default=20)

    parser.add_argument('--prior_path', type=str, default="ckpt/gpt_chembl.pt") # "ckpt/gpt_zinc.pt" "ckpt/rnn/epoch5.pt"
    parser.add_argument('--vocab_path', type=str, default="ckpt/vocab.txt")
    parser.add_argument('--log_dir', type=str, default="log_new/")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    writer = SummaryWriter(args.log_dir + f"{args.task}_{args.target}/{args.run_name}/")
    writer.add_text("configs", str(args))

    RL_trainer = RL_trainer(logger=writer, configs=args)
    RL_trainer.train()

    writer.close()
    