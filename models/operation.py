import torch
from torch import nn

class StatePoolLayer(nn.Module):
    def __init__(self, hidden_nf, N1, Nh):
        super(StatePoolLayer, self).__init__()
        # state attention model
        self.sam = nn.Sequential(
            nn.Linear(2*hidden_nf, hidden_nf),
            # nn.ELU(),
            # nn.Linear(hidden_nf, hidden_nf),
            nn.ELU(),
            nn.Linear(hidden_nf, 2*Nh),
        )

        # attention heads decoding
        self.zdm = nn.Sequential(
            nn.Linear(Nh * hidden_nf, hidden_nf),
            # nn.ELU(),
            # nn.Linear(hidden_nf, hidden_nf),
            nn.ELU(),
            nn.Linear(hidden_nf, N1),
        )

        # vector attention heads decoding
        self.zdm_vec = nn.Sequential(
            nn.Linear(Nh * hidden_nf, N1, bias=False)
        )

    def forward(self, q, p, M):   # q is the [num_atoms, 128]
        # create filter for softmax
        F = (1.0 - M + 1e-3) / (M - 1e-3)

        # pack features
        z = torch.cat([q, torch.norm(p, dim=1)], dim=1) # after sam.formward function we have z[num_atoms, hidden_nf]

        # print(f'\namong the pooling, F is nan: {torch.isnan(F).any()} and M: {torch.isnan(M).any()} and z isnan: {torch.isnan(z).any()} and q isnan:{torch.isnan(q).any()} P isnan: {torch.isnan(p).any()}')
        # multiple attention pool on state
        Ms = nn.functional.softmax(self.sam.forward(z).unsqueeze(1) + F.unsqueeze(2), dim=0).view(M.shape[0], M.shape[1], -1, 2)
        qh = torch.matmul(torch.transpose(q,0,1), torch.transpose(Ms[:,:,:,0],0,1))
        ph = torch.matmul(torch.transpose(torch.transpose(p,0,2),0,1), torch.transpose(Ms[:,:,:,1],0,1).unsqueeze(1))

        # print(f'\namong the pooling, Ms shape: {Ms.shape} and qh shape: {qh.shape}')
        # attention heads decoding
        qr = self.zdm.forward(qh.view(Ms.shape[1], -1))   # qr[num_residues, N1]
        pr = self.zdm_vec.forward(ph.view(Ms.shape[1], p.shape[1], -1))

        return qr, pr