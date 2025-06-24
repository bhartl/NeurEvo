import unittest


class TestEvolutionaryStrategyOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.cartpole_path = 'test/data/mindcraft/adapt/es_method/cart_pole.log'

        self.cartpole_env = dict(cls='GymWrapper', locate='mindcraft.envs.gym_wrapper', gym_id='CartPole-v1')

        self.cartpole_agent = dict(cls='TorchAgent',
                                   action_space='Box([0, 0], [+1, +1], dtype=np.float32)',
                                   default_action='[1, 0]',
                                   policy_module=dict(cls="Recurrent", layer_type="LSTM",
                                                      input_size=4, hidden_size=4, action_size=2,
                                                      ),
                                   )

        self.cartpole = dict(env=self.cartpole_env,
                             agent=self.cartpole_agent,
                             verbose=False,
                             render=False,
                             )

    @unittest.skip
    def test_cart_pole_no_checkpoint(self, n_episodes=3, size=16, max_steps=50, es_util='CMAES', verbose=False):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from mindcraft.train import EvolutionaryStrategy
            world = {k: v for k, v in self.cartpole.items()}
            world['n_episodes'] = n_episodes
            optimizer = EvolutionaryStrategy(world=self.cartpole,
                                             rollout_kwargs={'verbose': False, },
                                             rollout_aggregation='mean',
                                             population_size=size,
                                             es_util=es_util,
                                             x0=None,
                                             log_file=self.cartpole_path,
                                             checkpoint_interval=-1,
                                             )

            result = optimizer.run(max_steps=max_steps, verbose=verbose)
            if verbose:
                print(result)
