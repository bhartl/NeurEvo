from unittest import TestCase, skip


class TestKeyboardAgent(TestCase):
    def test_import(self, verbose=False):
        try:
            from mindcraft.agents import KeyboardAgent
            if verbose:
                print(KeyboardAgent)
        except ImportError:
            import warnings
            warnings.warn("Could not import `KeyboardAgent`. This could be rooted in not available XServer, e.g., on a cluster.")

            import traceback
            traceback.print_exc()

    @skip
    def test_car_racing(self, verbose=False, render=False):
        from mindcraft import World
        from mindcraft.agents import KeyboardAgent

        agent = KeyboardAgent(action_space='Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32)',
                              key_map={'Key.up': [0., 1., 0.],
                                       'Key.left': [-1., 0., 0.],
                                       'Key.right': [1., 0., 0.],
                                       'Key.down': [0., 0., 1.],
                                       },
                              default_action=[0., 0., 0.],
                              )

        env_repr = dict(cls='CarRacing',
                        locate='mindcraft.envs.gym_wrapper',
                        verbose=False,
                        gym_kwargs=dict(
                            render_mode='human',
                        ),
                        )

        world_repr = dict(env=env_repr,
                          agent=agent,
                          n_episodes=1,
                          load_steps=0,
                          verbose=verbose,
                          render=render,
                          delay=0.025,
                          )

        log_history = World.make_rollouts(world=world_repr, verbose=verbose)
