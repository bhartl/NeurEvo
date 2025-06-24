from mindcraft.agent import Agent
from mindcraft.io.spaces import Space
import threading
import numpy as np
from pynput import keyboard


class KeyboardAgent(Agent):
    """ A Keyboard-Controlled Agent - Define a key-map for your actions, and you are ready to go.

    (c) B. Hartl 2021
    """

    REPR_FIELDS = ('key_map', 'terminal_key', *Agent.REPR_FIELDS, )

    def __init__(self, key_map: dict, terminal_key: str = 'Key.esc', **kwargs):
        super(KeyboardAgent, self).__init__(**kwargs)

        self.key_map = key_map
        self.terminal_key = str(terminal_key)

        self._key_listener = None
        self._keyboard_listener = None
        self._keys = set()
        self.has_terminated = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate(self.terminal_key)

    def forward(self, observation: Space = None, reward=None, info=None) -> object:

        if not self._key_listener:
            self._key_listener = threading.Thread(name='key_listener', target=self.key_listener, daemon=True)
            self._key_listener.start()

        action = self.get_default_action()
        keys = self.get_keys()
        for key in keys:
            if self.terminate(key):
                return self.get_default_action()

            action += self.key_map.get(key, self.get_default_action())

        return action

    def get_default_action(self) -> np.ndarray:
        return np.array(self.default_action)

    def get_parameters(self):
        return None

    def set_parameters(self, parameters):
        pass

    def key_listener(self):
        self.has_terminated = False
        with keyboard.Listener(on_press=self.on_press,  on_release=self.on_release) as self._keyboard_listener:
            self._keyboard_listener.join()

    def on_press(self, key):
        if not self.has_terminated:
            key = self.format_key(key)
            self._keys.add(key)

    def on_release(self, key):
        key = self.format_key(key)

        try:
            self._keys.remove(key)
        except KeyError:
            pass

        return not self.terminate(key)

    @staticmethod
    def format_key(key):
        if isinstance(key, int):
            key = str(chr(key))

        elif key is not None:
            assert isinstance(key, (str, keyboard.Key)), key
            key = str(key)

            if key.startswith("\'"):
                key = key[1:]
            if key.endswith("\'"):
                key = key[:-1]

        return key

    def get_keys(self):
        if self.has_terminated:
            return set()

        return self._keys

    def terminate(self, key=None):
        if str(key) == self.terminal_key and not self.has_terminated:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            finally:
                self._keyboard_listener = None

            try:
                self._key_listener.join(1e-3)
            except Exception:
                pass
            finally:
                self._key_listener = None

            self.has_terminated = True

        return self.has_terminated
