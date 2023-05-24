import torch as th
import pdb

def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret

def build_td_lambda_targets_with_value(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda, relay, pos=None):
    relay = relay.unsqueeze(-1).to('cuda')
    ret = target_qs.new_zeros(*target_qs.shape).to('cuda')

    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    
    if pos is None:
        return ret
    else:
        # get unique p in pos
        roots = list(set(pos))
        ensemble_ret = th.zeros_like(ret).to('cuda')
        for r in roots:
            if r == -1:
                continue
            idx = [idx for idx, p in enumerate(pos) if p != r]

            e_relay = relay.clone()
            e_relay[idx, :] = 0
            relay_expand = th.zeros_like(ret).to('cuda')
            relay_expand[:, :e_relay.size(1)] = e_relay

            temp = (e_relay * ret[:, :e_relay.size(dim=1)]).sum(dim=0) / e_relay.sum(dim=0)
            e_relay = e_relay * temp

            inter_expand = th.zeros_like(ret).to('cuda')
            inter_expand[:, :e_relay.size(1)] = e_relay


            #pdb.set_trace()
            ensemble_ret += th.where(relay_expand == 1, inter_expand, 0)

        ret = th.where(ensemble_ret==0, ret, ensemble_ret)

        return ret