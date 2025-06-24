from unittest import TestCase
import numpy as np
import itertools as it

class TestSensoryNeuronAgent(TestCase):
    def test_init(self):
        from mindcraft.agents.sensory_neuron_agent import SensoryNeuronAgent

        from mindcraft.torch.module import FeedForward
        policy_module = FeedForward(
            input_size=2,
            output_size=1,
            hidden_size=[32, ] * 1,
            activation="Tanh",
        )

        agent = SensoryNeuronAgent(policy_module=policy_module,
                                   action_space="Box(-1, 1, (1,), dtype=np.float32)",
                                   )

        self.assertIsNotNone(agent)

    def test_dump_reload(self):
        from mindcraft.agents.sensory_neuron_agent import SensoryNeuronAgent

        from mindcraft.torch.module import FeedForward
        policy_module = FeedForward(
            input_size=2,
            output_size=1,
            hidden_size=[32, ] * 1,
            activation="Tanh",
        )

        agent = SensoryNeuronAgent(policy_module=policy_module,
                                   action_space="Box(-1, 1, (1,), dtype=np.float32)",
                                   )

        agent2 = SensoryNeuronAgent.make(agent.to_dict())
        self.assertIsNotNone(agent2)

        # Check if the two agents are equal
        params_agent = agent.get_parameters()
        params_agent2 = agent2.get_parameters()
        self.assertEqual(len(params_agent), len(params_agent2))
        self.assertTrue(np.array_equal(params_agent, params_agent2))

    def test_io(self):
        from torch import Tensor
        from mindcraft.torch.util import get_n_params
        from mindcraft.torch.module import SetTransformer
        from mindcraft.torch.module import SensoryEmbedding
        from mindcraft.torch.module import Recurrent
        from mindcraft.torch.module import Projection, LinearP
        from mindcraft.agents.sensory_neuron_agent import SensoryNeuronAgent

        num_features = 10
        num_channels = 3
        action_size = 3

        batch_size = 1
        key_channels = num_channels + action_size

        key_embed = 3
        val_embed = 3
        query_embed = 3
        val_size = 1
        context_size = 9

        channels_first = False
        np.random.seed(18648)

        for a, n, p, s in it.product(['Tanh', 'ReLU'],
                                     ['Identity', 'LayerNorm'],
                                     [None, LinearP, ],
                                     ['LSTM', 'RNN', 'GRU', None],  # 'NCP'],
                                     ):

            if p is None:
                if s is None:
                    continue

                s = Recurrent(input_size=key_channels, output_size=key_embed, layer_type=s, num_layers=2, is_nested=True) if s else None
                v = LinearP(input_size=num_channels, projection_size=val_embed, is_nested=True)
            else:
                # Recurrent(input_size=key_embed, layer_type=s, is_nested=True) if s else None
                p = p(input_size=key_channels, projection_size=key_embed, is_nested=True, )
                v = LinearP(input_size=num_channels, projection_size=val_embed, is_nested=True, )

            key_embedding = SensoryEmbedding(projection=p, sensor=s, is_nested=True)
            val_embedding = SensoryEmbedding(projection=v, is_nested=True)
            head = LinearP(context_size * val_size, action_size, is_nested=True)
            sensor = SetTransformer(seq_len=num_features,
                                    input_size=num_channels,
                                    key_embed=key_embedding,
                                    val_embed=val_embedding,
                                    qry_size=query_embed,
                                    context_size=context_size,
                                    val_size=val_size,
                                    qkv_bias=True,
                                    activation=a,
                                    norm_layer=n,
                                    head=head,
                                    channels_first=channels_first,
                                    disable_pos_embed=True,
                                    retain_grad=False,
                                    )

            agent = SensoryNeuronAgent(policy_module=sensor,
                                       action_space=f"Box(-1, 1, ({action_size},), dtype=np.float32)",
                                       default_action=0.,
                                       foldback_attrs=("action", ),
                                       )

            agent_copy = SensoryNeuronAgent.make(agent.to_dict())

            for _ in range(4):
                if channels_first:
                    x = (Tensor(np.random.rand(batch_size, num_channels, num_features)) - 0.5)
                else:
                    x = (Tensor(np.random.rand(batch_size, num_features, num_channels)) - 0.5)

                action = agent.get_action(observation=x)
                a_copy = agent_copy.get_action(observation=x)

                # Check if the two agents are equal
                self.assertTrue(np.array_equal(action, a_copy))
