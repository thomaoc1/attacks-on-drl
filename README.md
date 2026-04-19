# attacks-on-drl
Framework for existing attacks on DRL algorithms.

## Example Usage
```python
from attacks_on_drl.attacker import FGSMAttacker
from attacks_on_drl.runner import AttackRunner
from attacks_on_drl.victim import DQNVictim

# Defined environment (env) and SB3 policy (policy)
victim = DQNVictim(policy)
attacker = FGSMAttacker(victim)
runner = AttackRunner(env, attacker, victim, episode_max_frames=10_000)

runner.run(n_episodes=10)
```

## Implemented Attacks

1. FGSM $\ell_\infty$ Attacker [1]
2. Value Function Attack [2]
3. Strategically Timed Attack [3]
4. Critical Point Attack [4]

## References

[1] Huang, S., Papernot, N., Goodfellow, I., Duan, Y. and Abbeel, P., 2017. Adversarial attacks on neural network policies. arXiv preprint arXiv:1702.02284.

[2] Kos, J. and Song, D., 2017. Delving into adversarial attacks on deep policies. arXiv preprint arXiv:1705.06452.

[3] Lin, Y.C., Hong, Z.W., Liao, Y.H., Shih, M.L., Liu, M.Y. and Sun, M., 2017. Tactics of adversarial attack on deep reinforcement learning agents. arXiv preprint arXiv:1703.06748.

[4] Sun, J., Zhang, T., Xie, X., Ma, L., Zheng, Y., Chen, K. and Liu, Y., 2020, April. Stealthy and efficient adversarial attacks against deep reinforcement learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 04, pp. 5883-5891).
