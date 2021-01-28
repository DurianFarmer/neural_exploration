import numpy as np
import abc
from tqdm import tqdm

from .utils import inv_sherman_morrison_iter

class TS(abc.ABC):
    """Base class for TS methods.
    """
    def __init__(self,
                 bandit,
                 reg_factor=1.0,
                 confidence_scaling_factor=-1.0,
                 exploration_variance=1.0,
                 delta=0.1,
                 train_every=1,
                 throttle=int(1e2),
                ):
        # bandit object, contains features and generated rewards
        self.bandit = bandit
        # L2 regularization strength
        self.reg_factor = reg_factor
        # Confidence bound with probability 1-delta
        self.delta = delta
        # multiplier for the confidence bound (default is bandit reward noise std dev)
        if confidence_scaling_factor == -1.0:
            confidence_scaling_factor = bandit.noise_std
        self.confidence_scaling_factor = confidence_scaling_factor
        # exploration variance for TS
        self.exploration_variance = exploration_variance
        
        # train approximator only every few rounds
        self.train_every = train_every
        
        # throttle tqdm updates
        self.throttle = throttle
        
        self.reset()
    
    ## for TS
    def reset_sample_rewards(self):
        """Initialize sample rewards and related quantities.
        """
        self.sigma_square = np.empty((self.bandit.T, self.bandit.n_arms))
        self.mu_hat = np.empty((self.bandit.T, self.bandit.n_arms)) 
        self.sample_rewards = np.zeros((self.bandit.T, self.bandit.n_arms, self.bandit.n_samples))
        self.optimistic_sample_rewards = np.zeros((self.bandit.T, self.bandit.n_arms))
    
    ## for UCB
    def reset_upper_confidence_bounds(self):
        """Initialize upper confidence bounds and related quantities.
        """
        self.exploration_bonus = np.empty((self.bandit.T, self.bandit.n_arms))
        self.mu_hat = np.empty((self.bandit.T, self.bandit.n_arms)) 
        self.upper_confidence_bounds = np.ones((self.bandit.T, self.bandit.n_arms))
        
    def reset_regrets(self):
        """Initialize regrets.
        """
        self.regrets = np.empty(self.bandit.T)

    def reset_actions(self):
        """Initialize cache of actions (playing super arms).
        """
        ## --
        self.actions = np.empty((self.bandit.T, self.bandit.n_assortment)).astype('int')        
    
    def reset_A_inv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_inv = np.eye(self.approximator_dim)/self.reg_factor        
    
    def reset_grad_approx(self):
        """Initialize the gradient of the approximator w.r.t its parameters.
        """
        self.grad_approx = np.zeros((self.bandit.n_arms, self.approximator_dim))

    ## for TS
    def sample_action_ts(self):
        """Return the action (super arm) to play based on current estimates
        """
        ## --
        a = self.optimistic_sample_rewards[self.iteration]
        ind = np.argpartition(a, -1*self.bandit.n_assortment)[-1*self.bandit.n_assortment:]
        s_ind = ind[np.argsort(a[ind])][::-1].astype('int')
        return s_ind
    
    ## for UCB
    def sample_action(self):
        """Return the action (super arm) to play based on current estimates
        """        
        ## --
        a = self.upper_confidence_bounds[self.iteration]        
        ind = np.argpartition(a, -1*self.bandit.n_assortment)[-1*self.bandit.n_assortment:]
        s_ind = ind[np.argsort(a[ind])][::-1].astype('int')
        return s_ind                

    @abc.abstractmethod
    def reset(self):
        """Initialize variables of interest.
        To be defined in children classes.
        """
        pass

    @property
    @abc.abstractmethod
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        pass
    
    @property
    @abc.abstractmethod
    def confidence_multiplier(self):
        """Multiplier for the confidence exploration bonus.
        To be defined in children classes.
        """
        pass
    
    @abc.abstractmethod
    def update_confidence_bounds(self):
        """Update the confidence bounds for all arms at time t.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def update_output_gradient(self):
        """Compute output gradient of the approximator w.r.t its parameters.
        """
        pass
    
    @abc.abstractmethod
    def train(self):
        """Update approximator.
        To be defined in children classes.
        """
        pass
    
    @abc.abstractmethod
    def predict(self):
        """Predict rewards based on an approximator.
        To be defined in children classes.
        """
        pass
    
    ## for TS
    def update_sample_rewards(self):
        """Update sample rewards and related quantities for all arms.
        """        
        # update self.grad_approx
        self.update_output_gradient() 
        
        # update sigma_square        
        self.sigma_square[self.iteration] = [self.reg_factor * \
                                             np.dot(self.grad_approx[a], np.dot(self.A_inv, self.grad_approx[a].T)) \
                                             for a in self.bandit.arms]
                
        # update reward prediction mu_hat
        self.predict()
        
        # update sample reward
        self.sample_rewards[self.iteration] = [np.random.normal(loc = self.mu_hat[self.iteration, a], \
                                                                scale = self.exploration_variance * self.sigma_square[self.iteration, a], \
                                                                size = self.bandit.n_samples) \
                                               for a in self.bandit.arms]        
        
        # update optimistic sample reward for each arm
        self.optimistic_sample_rewards[self.iteration] = np.max(self.sample_rewards[self.iteration], axis=-1)

    ## for UCB
    def update_confidence_bounds(self):
        """Update confidence bounds and related quantities for all arms.
        """
        self.update_output_gradient()
        
        # TS exploration bonus
        self.exploration_bonus[self.iteration] = np.array(
            [
                self.confidence_multiplier * np.sqrt(np.dot(self.grad_approx[a], np.dot(self.A_inv, self.grad_approx[a].T))) for a in self.bandit.arms
            ]
        )        
        
        # update reward prediction mu_hat
        self.predict()
        
        # estimated combined bound for reward
        self.upper_confidence_bounds[self.iteration] = self.mu_hat[self.iteration] + self.exploration_bonus[self.iteration]
        
    def update_A_inv(self):
        ##
        self.A_inv = inv_sherman_morrison_iter(
            self.grad_approx[self.action],
            self.A_inv
        )               
    
    ## for TS
    def run(self):
        """Run an episode of bandit.
        """
        postfix = {
            'total regret': 0.0,
            '% optimal super arm': 0.0,
        }
        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):                
                ## update sample rewards of all arms based on observed features at time t
                self.update_sample_rewards()
                ## update confidence of all arms based on observed features at time t
                ## self.update_confidence_bounds()
                
                # pick action (super arm) with the highest boosted estimated reward
                self.action = self.sample_action_ts() ##
                self.actions[t] = self.action
                # update approximator
                if t % self.train_every == 0:
                    self.train()
                # update exploration indicator A_inv
                self.update_A_inv()
                
                ## compute regret
                self.regrets[t] = self.bandit.best_round_reward[t] - self.bandit.round_reward_function(self.bandit.rewards[t, self.action]) 
                
                # increment counter
                self.iteration += 1
                
                # log
                postfix['total regret'] += self.regrets[t]
                n_optimal_arm = np.sum(
                    np.prod(
                        (self.actions[:self.iteration]==self.bandit.best_super_arm[:self.iteration])*1, 
                        axis=1)                                                          
                )
                postfix['% optimal super arm'] = '{:.2%}'.format(n_optimal_arm / self.iteration)
                
                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)                      