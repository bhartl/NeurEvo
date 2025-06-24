from unittest import TestCase
import numpy as np


class TestEnsemble(TestCase):
    def test_feed_forward_ensemble(self, verbose=False):
        from mindcraft.torch.module import Ensemble
        from mindcraft.torch.module import FeedForward
        from mindcraft.torch.util import tensor_to_numpy
        from torch import randn

        ff = FeedForward(input_size=5, output_size=5)
        ensemble = Ensemble(nn=ff, redundancy=10, randomize=True, hooks=(), foo="mean")

        x = randn(2, 5)
        y = ensemble(x)
        if verbose:
            print("ensemble output")
            print(y)

        if verbose:
            print("ensemble.nn_0 parameters")
            print(ensemble.nn_0.serialize_parameters())
            print("ensemble.nn_1 parameters")
            print(ensemble.nn_1.serialize_parameters())
            print("ensemble parameters")
            print(ensemble.serialize_parameters())

        p = np.concatenate([getattr(ensemble, nn_label).serialize_parameters(to_numpy=True) for nn_label in ensemble.nn_labels])
        self.assertTrue(np.array_equal(p, ensemble.serialize_parameters(to_numpy=True)))

        reloaded = Ensemble.make(ensemble.to_dict())
        yr = reloaded(x)

        if verbose:
            print("reloaded ensemble output")
            print(yr)

        self.assertTrue(np.array_equal(tensor_to_numpy(y), tensor_to_numpy(yr)))

    def test_recurrent_ensemble_hook_states(self, verbose=False):
        from mindcraft.torch.module import Ensemble
        from mindcraft.torch.module import Recurrent
        from mindcraft.torch.util import tensor_to_numpy
        from torch import randn

        ff = Recurrent(input_size=5, hidden_size=2, num_layers=2, output_size=5)
        ensemble = Ensemble(nn=ff, redundancy=10, randomize=True, hooks=("states",), foo="mean")

        x = randn(2, 5)
        y = ensemble(x)
        if verbose:
            print("ensemble output")
            print(y)

        p = np.concatenate([getattr(ensemble, nn_label).serialize_parameters(to_numpy=True) for nn_label in ensemble.nn_labels])
        self.assertTrue(np.array_equal(p, ensemble.serialize_parameters(to_numpy=True)))

        reloaded = Ensemble.make(ensemble.to_dict())
        yr = reloaded(x)

        if verbose:
            print("reloaded ensemble output")
            print(yr)

        self.assertTrue(np.array_equal(tensor_to_numpy(y), tensor_to_numpy(yr)))

        if verbose:
            print("")
            print(ensemble.states)
            print(reloaded.states)

        self.assertIsNot(ensemble.states, reloaded.states)
        for esi, rsi in zip(ensemble.states, reloaded.states):
            self.assertIsNot(esi, rsi)
            self.assertTrue(np.array_equal(tensor_to_numpy(esi), tensor_to_numpy(rsi)))

    def test_recurrent_ensemble_unhook_states(self, verbose=False):
        from mindcraft.torch.module import Ensemble
        from mindcraft.torch.module import Recurrent
        from mindcraft.torch.util import tensor_to_numpy
        from torch import randn

        ff = Recurrent(input_size=5, hidden_size=2, num_layers=2, output_size=5)
        ensemble = Ensemble(nn=ff, redundancy=10, randomize=True, hooks=(), foo="mean")

        x = randn(2, 5)
        y = ensemble(x)
        if verbose:
            print("ensemble output")
            print(y)

        p = np.concatenate([getattr(ensemble, nn_label).serialize_parameters(to_numpy=True) for nn_label in ensemble.nn_labels])
        self.assertTrue(np.array_equal(p, ensemble.serialize_parameters(to_numpy=True)))

        reloaded = Ensemble.make(ensemble.to_dict())
        yr = reloaded(x)

        if verbose:
            print("reloaded ensemble output")
            print(yr)

        self.assertTrue(np.array_equal(tensor_to_numpy(y), tensor_to_numpy(yr)))

        if verbose:
            print("")
            print(ensemble.states)
            print(reloaded.states)

        self.assertIsNot(ensemble.states, reloaded.states)
        states = ensemble.states
        for esi, rsi in zip(states, reloaded.states):
            self.assertIsNot(esi, rsi)
            self.assertTrue(np.array_equal(tensor_to_numpy(esi), tensor_to_numpy(rsi)))

        for j, nn in enumerate(ensemble.nn_stack):
            states_j = nn.states
            for esi, esj in zip(states, states_j):
                esi_j = esi[j * nn.num_layers:(j+1) * nn.num_layers]
                self.assertTrue(np.array_equal(tensor_to_numpy(esi_j), tensor_to_numpy(esj)))
