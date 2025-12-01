import numpy as np
import gymnasium as gym
from gymnasium import spaces


def compute_pH_weak_acid_titration(
    Va_L: float,
    Ca: float,
    Vb_L: float,
    Cb: float,
    pKa: float,
) -> float:
    """
    Compute pH for titration of a monoprotic weak acid HA with strong base.

    Va_L: initial acid volume (L)
    Ca:   acid concentration (M)
    Vb_L: base volume added so far (L)
    Cb:   base concentration (M)
    pKa:  acid pKa
    """
    Kw = 1e-14
    Ka = 10 ** (-pKa)
    n_HA0 = Ca * Va_L            # initial moles of HA
    n_OH = Cb * Vb_L             # moles of OH- added
    Vtot = Va_L + Vb_L

    if Vtot <= 0:
        return 7.0

    # Before equivalence: buffer region
    if n_OH < n_HA0 - 1e-12:
        n_A = n_OH
        n_HA = n_HA0 - n_OH
        # avoid log(0)
        n_A = max(n_A, 1e-16)
        n_HA = max(n_HA, 1e-16)
        ratio = n_A / n_HA
        pH = pKa + np.log10(ratio)

    # At equivalence: solution of A- only (weak base)
    elif abs(n_OH - n_HA0) <= 1e-12:
        C_A = n_HA0 / Vtot
        Kb = Kw / Ka
        OH = np.sqrt(max(Kb * C_A, 1e-20))
        pOH = -np.log10(OH)
        pH = 14.0 - pOH

    # After equivalence: excess strong base
    else:
        n_excess = n_OH - n_HA0
        OH = n_excess / Vtot
        OH = max(OH, 1e-20)
        pOH = -np.log10(OH)
        pH = 14.0 - pOH

    return float(np.clip(pH, 0.0, 14.0))


def indicator_rgb_from_pH(
    pH: float,
    pKa_ind: float = 7.0,
    neutral_band: float = 0.15,
) -> np.ndarray:
    """
    Simulate an acid/base indicator with a continuous color response.

    - pKa_ind: indicator transition midpoint
    - neutral_band: range around pH ~7 where color is strongly "neutral"
    """
    # Acid form color (e.g. yellow) and base form (e.g. blue)
    acid_rgb = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # yellow
    base_rgb = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # blue
    neutral_rgb = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # green

    # Fraction of base-colored form (In-) from indicator acid-base equilibrium
    # f_base = 1 / (1 + 10^(pKa_ind - pH))
    f_base = 1.0 / (1.0 + 10 ** (pKa_ind - pH))
    f_base = float(np.clip(f_base, 0.0, 1.0))
    f_acid = 1.0 - f_base

    # Base acid/base blend
    base_mix = f_acid * acid_rgb + f_base * base_rgb

    # Strength of neutral influence: strongest near pH_target = 7 +/- neutral_band
    pH_target = 7.0
    dist = abs(pH - pH_target)
    if dist >= neutral_band:
        # Outside neutral band: just use acid/base mix
        return base_mix

    # Inside neutral band: smoothly mix in neutral color.
    # weight goes 1 at pH_target, 0 at edge of band.
    w = 1.0 - dist / neutral_band
    rgb = (1 - w) * base_mix + w * neutral_rgb
    return rgb.astype(np.float32)


class WeakAcidIndicatorEnv(gym.Env):
    """
    Gym environment for titrating a weak acid with strong base,
    where the agent *sees* only indicator color + volume ratio + step,
    not the true pH.

    Observation:
        [R, G, B, Vb_over_Veq, step_norm]

    Actions:
        0: add small volume of base
        1: add medium volume
        2: add large volume
        3: stop

    Reward:
        -|pH - pH_target| / 7  - 0.01 per step,
        plus bonus when stopping close to target.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        Va_ml: float = 50.0,
        Ca: float = 0.1,
        Cb: float = 0.1,
        pKa: float = 4.76,    # acetic acid-like
        pH_target: float = 7.0,
        max_steps: int = 40,
        step_sizes_ml=(0.1, 0.5, 1.0),
        pKa_ind: float = 7.0,
        neutral_band: float = 0.15,
    ):
        super().__init__()

        self.Va_ml = Va_ml
        self.Ca = Ca
        self.Cb = Cb
        self.pKa = pKa
        self.pH_target = pH_target
        self.max_steps = max_steps
        self.step_sizes_ml = np.array(step_sizes_ml, dtype=float)
        self.pKa_ind = pKa_ind
        self.neutral_band = neutral_band

        # Derived constants
        self.Va_L = Va_ml / 1000.0
        self.n_HA0 = self.Ca * self.Va_L
        self.Vb_L = 0.0
        self.step_count = 0

        # equivalence volume:
        self.Veq_L = self.n_HA0 / self.Cb

        # Observation: [R, G, B, Vb/Veq, step_norm]
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Actions: 0..len(step_sizes) = add base; last index = stop
        self.action_space = spaces.Discrete(len(self.step_sizes_ml) + 1)

        self.history = []

    def _get_pH(self) -> float:
        return compute_pH_weak_acid_titration(
            Va_L=self.Va_L,
            Ca=self.Ca,
            Vb_L=self.Vb_L,
            Cb=self.Cb,
            pKa=self.pKa,
        )

    def _get_obs(self, pH: float) -> np.ndarray:
        rgb = indicator_rgb_from_pH(
            pH,
            pKa_ind=self.pKa_ind,
            neutral_band=self.neutral_band,
        )

        V_ratio = self.Vb_L / self.Veq_L
        step_norm = self.step_count / self.max_steps

        return np.concatenate([rgb, [V_ratio, step_norm]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.Vb_L = 0.0
        self.step_count = 0
        self.history = []
        pH = self._get_pH()
        obs = self._get_obs(pH)
        self.history.append((self.Vb_L * 1000.0, pH, obs[:3]))  # store (mL, pH, color)

        return obs, {}

    def step(self, action: int):
        self.step_count += 1
        terminated = False
        truncated = False

        # stop action: last index
        if action == self.action_space.n - 1:
            terminated = True
        else:
            delta_ml = self.step_sizes_ml[action]
            self.Vb_L += delta_ml / 1000.0

        pH = self._get_pH()
        self.history.append((self.Vb_L * 1000.0, pH, indicator_rgb_from_pH(pH, self.pKa_ind, self.neutral_band)))

        # base reward: closeness to target pH + small step penalty
        dist = abs(pH - self.pH_target)
        reward = -dist / 7.0 - 0.01

        if terminated:
            if dist < 0.05:
                reward += 1.0
            elif dist < 0.2:
                reward += 0.5

        # truncate if max steps or extreme conditions
        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        if pH <= 0.0 or pH >= 14.0 or self.Vb_L > 2.0 * self.Veq_L:
            terminated = True

        obs = self._get_obs(pH)
        info = {"pH": pH, "Vb_ml": self.Vb_L * 1000.0, "dist": dist}

        return obs, reward, terminated, truncated, info

    def render(self):
        # You can later implement a nice matplotlib plot of self.history
        pass

