def run(verbose=True, render=True):
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
                    verbose=verbose,
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Car Racing demo with Keyboard Agent")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('--render', action='store_true', help="Enable rendering of the environment")

    args = parser.parse_args()

    run(verbose=args.verbose, render=args.render)
