from TAMER.human_reward_model import HumanRewardModel
from torch.autograd import Variable
import torch
import torch.optim as optim
import math
import random


class experience:
    """
    Experience is an (x,y) pair, where x = (s,a,t,t+1) and y = (label,feedback-time)
    """

    def __init__(self, state, action, ts, te, tf, label):
        self.state = state
        self.action = action
        self.te = te
        self.ts = ts
        self.weight = 0.0
        self.tf = tf
        self.label = label


class TamerAgent:
    def __init__(self, env, gamma=0, lower_window_size=0.2, upper_window_size=4):

        self.gamma = gamma

        ## lower_window_size and upper_window_size are hyperparamters to be used while calcuating "importance weights"
        self.lower_window_size = lower_window_size
        self.upper_window_size = upper_window_size

        ## A list of non-zero weighted experiences for a single feedback (Dj)
        ## updated every time a new human signal is received
        self.experiences = []

        self.total_experiences = []
        self.x_list = []

        ## Initializing the human reward model H
        self.model = HumanRewardModel(env)

    def updateWeightsforSignal(self, signal, feedback_time):
        """
        Recieves human signal and updates the weights of all x (i.e assigns credit to them)
        according to some rule.
        Current function uses this rule from the paper: "each observed feedback y will only have nonzero w for those x observed between 4 and 0.2 seconds before the feedback occurred."
        The non-zero (x,y) pairs or experiences are added to 'experiences' list and the 'total_experiences' list is also updated
        """

        self.experiences = []

        ## assigning equal weight to every x within 0.2 to 4 secs before the feedback (not ideal!!)
        weight_per_experience = 1.0 / (self.upper_window_size - self.lower_window_size)

        for exp in self.x_list:
            # print(exp.te, exp.ts, feedback_time)
            if (exp.te < feedback_time - self.upper_window_size) or (
                exp.ts > feedback_time - self.lower_window_size
            ):
                continue
            else:
                exp.tf = feedback_time
                exp.label = signal
                exp.weight = weight_per_experience
                self.addExperience(exp)

        self.total_experiences.extend(self.experiences)

    def getAction(self, state):
        """
          Compute the best action to take in a state.
        """
        Q = self.model(Variable(torch.from_numpy(state).type(torch.FloatTensor)))
        maxQ1, max_action = torch.max(Q, -1)

        return max_action.item()

    def addExperience(self, experience):
        """
           Adding non-zero weighted experiences (for every human feedback received)
        """
        self.experiences.append(experience)

    def SGD_update(self, type, learning_rate=0.01, mini_batch_size=32):

        # train_on_gpu = torch.cuda.is_available()

        # if train_on_gpu:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)
        #     self.model = self.model.cuda()

        # print(self.model)

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        self.model.train()

        ## SGD updates are performed - 1) when a human feedback is received, 2) at fixed rate using the feedback buffer
        ## Using the feedback type to get the batch-size

        if type == "human":
            mini_batch = self.experiences
            batch_size = len(self.experiences)

        else:
            batch_size = min(len(self.total_experiences), mini_batch_size)
            mini_batch = random.sample(self.total_experiences, batch_size)

        # print(batch_size)
        loss = torch.FloatTensor([0.0])

        ## Using the loss fucntion from Deep TAMER paper:
        ## A weighted difference between the human reward and the predicted value for each
        ## state-action pair (not perfect!)
        for i in range(batch_size):
            weight = torch.FloatTensor([mini_batch[i].weight])
            state = torch.FloatTensor(mini_batch[i].state)
            action = mini_batch[i].action
            H_s = self.model(state)
            H_sa = H_s[action]
            label = torch.FloatTensor([mini_batch[i].label])

            individual_loss = weight * (H_sa - label).pow(2)
            loss = loss + individual_loss

        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
