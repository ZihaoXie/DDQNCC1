# Discussion

We believe that using a MainAgent which fingerprints the adversary and assigns a defending agent profile, is likely the best approach (assuming the agents it selects are “optimal”) due to the red agent’s behaviour not changing mid-episode and that the first few steps (used for fingerprinting) set as “Sleep” for the blue agent does not downgrade the performance. We considered informing the MainAgent once an episode ends (with agent.end_episode()), however we felt this would not accurately represent the challenge (as it was not in the original evaluation.py file) and as a result the fingerprinting does not use this information.

We also speculate that after fingerprinting the red agent, the best strategy for MainAgent may be to call an ensemble of different B_line and Meander blue agent models. However, given that we implemented a single approach, this was not possible.

To fingerprint the red agent, we sum the past two 52-bit observations and hard-coded them into the MainAgent. We also added some memory to the MainAgent so that in the rare case where it fingerprints Sleep when it shouldn't, it remembers the previous agent it assigned before the sleep (if the sum over the observation vector is greater than 0, i.e activity is visible) and reverts to it if this occurred in the past 3 steps.

        sleep_fingerprinted = [0] * 52
        meander_fingerprinted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bline_fingerprinted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bline_fingerprinted_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
Finally, for the B_line blue agent, we reduced the possible random actions in our epsilon greedy exploration to ones which may be useful. This significantly improved our training.

        def get_action(self, observation, action_space=None):
                if np.random.random() > self.epsilon:
                        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                        actions = self.q_eval.forward(state)
                        action = T.argmax(actions).item()
                else:
                        #action = np.random.choice(self.action_space)
                        possibly_useful_actions_bline = [0,1,4,5,9,10,11,17,18,22,23,24,30,31,35,36,37]
                        action = random.choice(possibly_useful_actions_bline)


Overall, we feel that version 1 served as a good introduction to the challenge, and we look forward to version 2.



# Results

Our results outperform BlueReactRestoreAgent and BlueReactRemoveAgent. This was checked in benchmarks.py

### evaluation.py on 1000 episodes, with random.seed(1) for reproducibility:

The full output of the evaluation is in MainAgent_1000episodes.txt.

*30 length episodes*
1. steps: 30, adversary: B_lineAgent, mean: -4.504999999999999, standard deviation 2.713916052559536
2. steps: 30, adversary: RedMeanderAgent, mean: -4.642, standard deviation 4.446842746310062
3. steps: 30, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

*50 length episodes*
1. steps: 50, adversary: B_lineAgent, mean: -8.309999999999995, standard deviation 4.62701948443156
2. steps: 50, adversary: RedMeanderAgent, mean: -9.245899999999999, standard deviation 5.935207652633555
3. steps: 50, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

*100 length episodes*
1. steps: 100, adversary: B_lineAgent, mean: -17.549999999999994, standard deviation 8.43009884678505
2. steps: 100, adversary: RedMeanderAgent, mean: -23.9663, standard deviation 15.566763261141613
3. steps: 100, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

### Total score: -68.2192
