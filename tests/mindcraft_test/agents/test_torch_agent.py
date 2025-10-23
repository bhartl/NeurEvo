from unittest import TestCase


class TestTorchAgent(TestCase):
    def test_rollout_gym(self, verbose=False, render=False, gym_id='LunarLander-v3', policy_module='Recurrent', ):
        """ Rollout a DirectAgent instance on any gym_id
          - Note that the Agent is randomly initialized

        :param gym_id: OpenAI Gym id, try 'CartPole-v0' or 'LunarLander-v2', for instance
                       (and see https://gym.openai.com/envs/).
        :param policy_module: Choose policy module from `mindcraft.torch.moduels` to
                             load a generic controller for the Agent, or pass a controller instance, defaults to 'CLSTMModel'
        """
        from mindcraft import World, Env
        from mindcraft.io.spaces import Discrete
        from mindcraft.torch import module

        env_repr = dict(cls='GymWrapper',
                        locate='mindcraft.envs',
                        gym_id=gym_id,
                        verbose=False,
                        )

        env = Env.make(env_repr)
        if verbose:
            env.details()

        assert len(env.observation_space.shape) == 1, "Only 1d data permitted. "

        if isinstance(policy_module, str):
            if policy_module == 'Recurrent':
                policy_module = module.Recurrent(input_size=env.observation_space.shape[0],
                                                 hidden_size=32,
                                                 stateful=True,
                                                 num_layers=2,
                                                 layer_type="LSTM",
                                                 output_size=env.action_space.n
                                                 )
            elif policy_module == 'FeedForward':
                policy_module = module.FeedForward(input_size=env.observation_space.shape[0],
                                                   inter_size=32,
                                                   command_size=5,
                                                   output_size=env.action_space.n
                                                   )

            else:
                raise NotImplementedError(policy_module)

        if isinstance(env.action_space, Discrete):
            n_actions = env.action_space.n
            action_space = f'Box({[0] * n_actions}, {[1] * n_actions}, dtype=np.float32)'
            default_action = [1.] + [0.] * (n_actions - 1)

        else:
            action_space = env.action_space
            default_action = [0.] * action_space.shape[0]

        agent_repr = dict(cls='TorchAgent',
                          locate='mindcraft.agents',  # could be omitted
                          action_space=action_space,
                          default_action=default_action,
                          policy_module=policy_module,
                          )

        world_repr = dict(env=env_repr,
                          agent=agent_repr,
                          n_episodes=8,
                          max_steps=1000,
                          render=render,
                          verbose=verbose,
                          )

        World.make_rollouts(world_repr, verbose=verbose)

    def test_serialize_reload(self, verbose=True):
        from mindcraft.agents import TorchAgent
        from mindcraft.torch.util import tensor_to_numpy
        from torch import Tensor
        from numpy import ndarray, array_equal
        from numpy.random import choice
        from itertools import product as it

        input_size = range(1, 21, 2)
        output_size = range(1, 201, 50)
        hidden_size = range(1, 21, 5)

        for i, o in it(input_size, output_size):
            for h in range(len(hidden_size) + 1):
                hs = None
                if h == 1:
                    hs = int(choice(hidden_size))
                elif h > 1:
                    hs = list(hidden_size)[0:h]

                t = TorchAgent(policy_module=dict(cls="FeedForward", input_size=i, hidden_size=hs, output_size=o),
                               action_space="Box(-1., 1., (2,))", default_action=[0., 0.],
                               )

                t.np_fields = True
                p = t.get_parameters()
                self.assertIs(type(p), ndarray)

                t1 = TorchAgent(policy_module=dict(cls="FeedForward", input_size=i, hidden_size=hs, output_size=o),
                                action_space="Box(-1., 1., (2,))", default_action=[0., 0.],
                                )
                t1.np_fields = False
                p1 = t1.get_parameters()
                self.assertIs(type(p1), Tensor)
                self.assertFalse(array_equal(p, tensor_to_numpy(p1)))  # random init

                t1.set_parameters(p)  # set params, should be equal now
                self.assertTrue(array_equal(p, tensor_to_numpy(t1.get_parameters())))

                t2 = TorchAgent.from_dict(t.to_dict())  # reload from dict
                self.assertTrue(array_equal(p, t2.get_parameters()))

                fname = "data/test/mindcraft_test/agents/test_torch_agent/test_serialize_reload.json"
                t.to_json(filename=fname)
                t3 = TorchAgent.make(fname)
                self.assertTrue(array_equal(p, t3.get_parameters()))
