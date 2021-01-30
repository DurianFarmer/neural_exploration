import numpy as np
from .ts import TS


class LinTS(TS):
    """Linear TS.
    """
    def __init__(self,
                 bandit,
                 reg_factor=1.0,
                 delta=0.01,
                 bound_theta=1.0,
                 confidence_scaling_factor=0.0,
                 throttle=int(1e2),
                ):

        # range of the linear predictors
        self.bound_theta = bound_theta
        
        # maximum L2 norm for the features across all arms and all rounds
        self.bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))

        super().__init__(bandit, 
                         reg_factor=reg_factor,
                         confidence_scaling_factor=confidence_scaling_factor,
                         delta=delta,
                         throttle=throttle,
                        )

    @property
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        return self.bandit.n_features
    
    def update_output_gradient(self):
        """For linear approximators, simply returns the features.
        """
        self.grad_approx = self.bandit.features[self.iteration]
    
    def reset(self):
        """Return the internal estimates
        """
        self.reset_upper_confidence_bounds()
        self.reset_sample_rewards() ##
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

        # randomly initialize linear predictors within their bounds
        ## --
        self.theta = np.random.uniform(-1, 1, self.bandit.n_features) * self.bound_theta
        ## self.theta = np.random.uniform(-1, 1, (self.bandit.n_arms, self.bandit.n_features)) * self.bound_theta # for disjoint model
        
        # initialize reward-weighted features sum at zero
        self.b = np.zeros(self.bandit.n_features)

    @property
    def confidence_multiplier(self):
        """LinTS confidence interval multiplier.
        """
        return self.confidence_scaling_factor \
    * np.sqrt(self.bandit.n_features*np.log((1+self.iteration*self.bound_features/self.reg_factor)/self.delta))\
    + np.sqrt(self.reg_factor)*np.linalg.norm(self.theta, ord=2)

    def train(self):
        """Update linear predictor theta.
        """
        ## --
        self.theta = np.matmul(self.A_inv, self.b)                      
        self.b += np.sum(np.array(\
                                  [ self.bandit.rewards[self.iteration][i]*self.bandit.features[self.iteration, self.action][i] \
                                   for i in range(0, self.bandit.n_assortment) ] \
                                 ), axis = 0)              
        
        ##self.theta = np.array(
        ##    [
        ##        np.matmul(self.A_inv[a], self.b[a]) for a in self.bandit.arms
        ##    ]
        ##)

        ## self.b += self.bandit.features[self.iteration, self.action]*self.bandit.rewards[self.iteration, self.action]
        
    
    def predict(self):
        """Predict reward.
        """
        self.mu_hat[self.iteration] = np.array(
            [
                np.dot(self.bandit.features[self.iteration, a], self.theta) for a in self.bandit.arms
            ]
        )