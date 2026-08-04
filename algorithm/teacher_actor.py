"""Pretrained teacher policy adapter for environment rollouts."""

from .algorithm import Algorithm


class TeacherActor(Algorithm):
    """Drive an environment with the MDP's exported teacher actor.

    The actor consumes the MDP's cached current observation.  Calling ``act``
    therefore does not advance observation histories; ``env.step`` advances
    them exactly once after physics.
    """

    def act(self, state=None):
        if state is None:
            state = self.env.get_state(self.env.all_envs_idx)
        return self.env.MDP.policy_action(**state)

    def train(self):
        raise RuntimeError("TeacherActor is fixed; use act() or eval() to collect demonstrations")

    def eval(self, steps: int):
        observations, actions, rewards, infos, mean_infos = [], [], [], [], []
        self.env.reset()
        for _ in range(steps):
            state = self.env.get_state(self.env.all_envs_idx)
            action = self.act(state)
            observation, reward, terminated, truncated, info = self.env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            mean_infos.append({
                key: value.detach().mean().cpu().item()
                if hasattr(value, "detach") else value
                for key, value in info.items()
            })
            if terminated.any() or truncated.any():
                break
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "infos": infos,
            "mean_infos": mean_infos,
        }
