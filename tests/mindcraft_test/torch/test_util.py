from unittest import TestCase


class TestUtil(TestCase):
    def test_auto_grad(self):
        from mindcraft.torch.util import AutoGrad
        from mindcraft.torch.module import FeedForward
        import torch

        batch_size = 10
        output_size = 1
        steps = 10

        device = 'cpu'  # 'cuda'  #
        for input_size in [5, 10]:
            model = FeedForward(input_size=input_size, output_size=output_size, device=device)
            x = torch.rand(batch_size, input_size, device=device) * 100.

            for lr in [1e-1, 1e-2]:
                for descent in [True, False]:
                    auto_grad = AutoGrad(model=model, steps=steps, lr=lr, descent=descent, device=device)

                    with torch.no_grad():
                        y = model(x)

                    x_grad, y_prime, _ = auto_grad(x, clone=True)

                    self.assertIsNot(x, x_grad)
                    with torch.no_grad():
                        if descent:
                            self.assertTrue(torch.all((y - y_prime) > 0.))
                        else:
                            self.assertTrue(torch.all((y - y_prime) < 0.))

    def test_remc(self):
        from mindcraft.torch.util import REMC
        from mindcraft.torch.util import tensor_to_numpy
        from mindcraft.torch.module import FeedForward
        import torch

        output_size = 1
        tempering_steps = 5
        sampling_steps = 10
        num_beta = 8

        device = 'cpu'  # 'cuda'
        for input_size in [5, 10]:
            model = FeedForward(input_size=input_size, output_size=output_size, device=device)
            model.eval()

            x = torch.rand(num_beta, input_size, device=device) * 10.
            remc = REMC(model=model, tempering_steps=tempering_steps, sampling_steps=sampling_steps,
                        n_samples=x.shape[0], n_params=x.shape[1], num_beta=num_beta, device=device,
                        )

            with torch.no_grad():
                y = model(x)
                x_tempered, y_prime, info = remc(x)
                self.assertIsNot(x, x_tempered)
                self.assertTrue(torch.all((y - y_prime) < 0.))


