import torch

class DummyEnv:
    def __init__(self):
        self.image_shape = (3, 64, 64)

    def get_observation(self):
        """Returns a random image tensor."""
        return torch.rand(self.image_shape)

    def step(self, action):
        """Prints the action taken."""
        print(f"Action taken: {action.argmax().item()}")
        # In a real environment, this would affect the state and return a new observation, reward, etc.
        return self.get_observation(), 0, False, {}
