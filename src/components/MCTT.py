import numpy as np
from cuml import KMeans
import torch as th
#from utils.kmeans import kmeans_predict
from multiprocessing import Pool

import pdb

class Node(object):

    def __init__(self, id, value):

        self.node_id = id
        self.children = []
        self.value = value
        self.father = None
        self.rewards = 0
        self.visit = 0
        self.depth = 0

    def cal_uct(self, c, test_mode=False, scale=None):

        if self.father == None:
            return 0
        else:
            if not scale:
                scale = 1

            if test_mode:
                return self.value
            f_visit = self.father.visit
            uct = self.value / max(scale, 0.001) + c * np.sqrt(np.log(f_visit) / self.visit)
            return uct

    def backprop(self, rewards, gamma=0.99):

        node = self
        gt = 0
        for i in range(len(rewards)-1, node.depth-1, -1):
            gt = gt * gamma + rewards[i]
        once = True
        while node != None:

            if once:
                node.rewards = (node.rewards * node.visit + gt) / (node.visit + 1)
                once = False
            else:
                node.rewards = (node.rewards * node.visit + rewards[node.depth]) / (node.visit + 1)

            node.value = sum([c.visit * (c.rewards + gamma * c.value) for c in node.children]) / (node.visit + 1)
            node.visit += 1
            node = node.father
        
    def __repr__(self):
        return str(self.value)

class Tree(object):

    def __init__(self, max_seq_length, predictor_sample, tree_sample):
        self.c = 0.02
        self.roots = []
        self.max_seq_length = min(max_seq_length, 50)

        self.cluster_num = 100
        self.sample_num = 2000

        self.predictor_sample = predictor_sample
        self.tree_sample = tree_sample
        self.warm_up = True
        self.scale = None

    def generate_predictor(self, replay_buffer, slow): 

        samples, index = replay_buffer.sample_by_index(self.sample_num)
        states = samples['state'][:, :self.max_seq_length].to('cuda')
        terminated = samples['terminated'][:, :self.max_seq_length].to('cuda')
        
        nodes = []
        for episode_idx in range(index):
            for tran_idx in range(self.max_seq_length):
                if terminated[episode_idx, tran_idx] == 1:
                    break
                nodes.append(states[episode_idx, tran_idx])

        nodes = th.stack(nodes)

        self.kmeans = KMeans(n_clusters=self.cluster_num, n_init=5, max_iter=100).fit(nodes)


    def predict(self, y):
        cluster_ids_y = self.kmeans.predict(y.to('cuda'))
        return cluster_ids_y

    
    def generate_tree_from_buffer(self, replay_buffer, slow=False):
        
        self.generate_predictor(replay_buffer, slow=slow)
        del self.roots
        self.roots = []

        samples, index = replay_buffer.sample_by_index(self.sample_num)
        
        states = samples['state'][:, :self.max_seq_length]     #(bs, t, state_size)
        terminated = samples['terminated'][:, :self.max_seq_length]
        rewards = samples['reward'][:, :]
        #states_id = self.predict(states.reshape(-1, states.size(dim=-1))).reshape(states.shape[:-1])
        '''
        states_id = []
        for s in states:
            s_id = self.predict(s.reshape(-1, s.size(dim=-1))).reshape(s.shape[:-1])
            states_id.append(s_id)
       
        states_id = th.stack(states_id)
        '''
        states_id = self.predict(states.reshape(-1, states.size(-1))).reshape(states.shape[:-1])
        for episode_idx in range(index):

            in_root = False
            for r in self.roots:
                if r.node_id == states_id[episode_idx, 0]:
                    n = r
                    in_root = True
                    break
            if in_root == False:
                n = Node(states_id[episode_idx, 0], 0)
                self.roots.append(n)

            for tran_idx in range(1, self.max_seq_length):
                if terminated[episode_idx, tran_idx] == 1:
                    break
                
                node_id = states_id[episode_idx, tran_idx]
                in_children = False

                for node in n.children:
                    if node_id == node.node_id:
                        n = node
                        in_children = True
                        break
                
                if in_children == False:
                    new_node = Node(node_id, 0)
                    new_node.father = n
                    new_node.depth = n.depth + 1
                    n.children.append(new_node)
                    n = new_node

            n.backprop([r[0] for r in rewards[episode_idx].tolist()])
        print('Finish Generating')

    def update_tree(self, ep_batch):

        states = ep_batch['state']
        rewards = ep_batch['reward']
        terminated = ep_batch['terminated']

        states_id = self.predict(states.reshape(-1, states.size(dim=-1)))   #(151, 1)
        
        in_root = False
        for r in self.roots:
            if r.node_id == states_id[0]:
                n = r
                in_root = True
                break
        if in_root == False:
            n = Node(states_id[0], 0)
            self.roots.append(n)

        for tran_idx in range(1, self.max_seq_length):
            if terminated[0][tran_idx] == 1:
                break
            
            node_id = states_id[tran_idx]

            in_children = False
            for node in n.children:
                if node_id == node.node_id:
                    n = node
                    in_children = True
                    break

            if in_children == False:
                new_node = Node(node_id, 0)
                new_node.father = n
                new_node.depth = n.depth + 1
                n.children.append(new_node)
                n = new_node
                
        n.backprop([r[0] for r in rewards[0].tolist()])



    def judge_templates(self, states, rewards):
        returns = rewards.sum(dim=-1).cpu().tolist()
        upper = np.percentile(returns, 95)
        lower = np.percentile(returns, 5)

        if self.scale is None:
            self.scale = upper - lower
        else:
            self.scale = self.scale * 0.9 + (upper - lower) * 0.1

        states_id = self.predict(states.reshape(-1, states.size(-1))).reshape(states.shape[:-1])
        relays = []
        

        positions = [-1 for _ in range(states_id.shape[0])]
        for idx, traj in enumerate(states_id):
            relay = []
            for r_idx, r in enumerate(self.roots):
                if r.node_id == traj[0]:
                    positions[idx] = r_idx
                    node = r
                    relay.append(1)
                    while len(node.children) != 0:
                        node = max([n for n in node.children], key=lambda x: x.cal_uct(self.c, scale=self.scale))
                        if node.node_id == traj[node.depth]:
                            relay.append(1)
                        else:
                            break
                    break
            relays.append(relay)
        
        max_length = max([len(r) for r in relays])

        for i in range(len(relays)):
            relays[i].extend([0 for _ in range(max_length - len(relays[i]))])        

        return th.tensor(relays).to('cuda'), positions
    


    def predict_one_state(self, state):
        state = th.tensor(state).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        
        return self.predict(state.reshape(-1, state.size(-1)))


    def generate_template(self, state):

        # Here state is numpy.ndarray
        state = th.tensor(state).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        
        state_id = self.predict(state.reshape(-1, state.size(-1)))
        in_root = False
        for r in self.roots:
            if r.node_id == state_id:
                node = r
                in_root = True
                break
        if not in_root:
            return []
        else:
            template = [node.node_id]
            while len(node.children) != 0:
                node = max([n for n in node.children], key=lambda x: x.cal_uct(self.c, self.scale))
                template.append(node.node_id)
            return template
