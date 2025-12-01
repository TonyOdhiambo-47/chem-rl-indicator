import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments


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
        # Special case: no base added yet, pure weak acid
        if n_OH <= 1e-12:
            # For weak acid: [H+] = sqrt(Ka * Ca)
            H_plus = np.sqrt(max(Ka * Ca, 1e-20))
            pH = -np.log10(H_plus)
        else:
            # Buffer region: Henderson-Hasselbalch
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

    metadata = {"render_modes": ["human", "rgb_array", "matplotlib"]}

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
        max_burette_ml: float = 50.0,  # Realistic burette capacity
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
        self.max_burette_ml = max_burette_ml  # Maximum volume from burette

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
        self._last_dist = None  # Track distance for progress bonus
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
            new_Vb_ml = (self.Vb_L * 1000.0) + delta_ml
            
            # REALISTIC CONSTRAINT: Burette maximum capacity
            if new_Vb_ml > self.max_burette_ml:
                # Can't add more - burette is empty
                terminated = True
                truncated = False
                # Heavy penalty for running out of titrant
                pH = self._get_pH()
                dist = abs(pH - self.pH_target)
                reward = 50.0 * np.exp(-dist / 0.8) - 0.005
                reward -= 30.0  # Penalty for running out
                obs = self._get_obs(pH)
                info = {"pH": pH, "Vb_ml": self.Vb_L * 1000.0, "dist": dist, "burette_empty": True}
                return obs, reward, terminated, truncated, info
            else:
                self.Vb_L += delta_ml / 1000.0

        pH = self._get_pH()
        self.history.append((self.Vb_L * 1000.0, pH, indicator_rgb_from_pH(pH, self.pKa_ind, self.neutral_band)))

        # base reward: closeness to target pH + small step penalty
        dist = abs(pH - self.pH_target)
        
        # CRITICAL: ASYMMETRIC REWARD - Heavy penalty for overshooting
        # Overshooting is MUCH worse than undershooting
        pH_overshoot = pH - self.pH_target  # Positive if above target
        
        # PORTFOLIO-GRADE REWARD SHAPING with anti-overshoot focus
        # Multi-component reward that guides learning through exploration
        
        # 1. Primary reward: Exponential closeness to target (very strong gradient)
        # At dist=0: reward = 50, at dist=7: reward ≈ 0.5
        reward = 50.0 * np.exp(-dist / 0.8) - 0.005  # Minimal step penalty
        
        # 2. ANTI-OVERSHOOT: EXTREME penalties for going above pH 7.0
        if pH > self.pH_target:
            # Progressive penalty that increases with overshoot - MUCH STRONGER
            overshoot_amount = pH - self.pH_target
            if overshoot_amount > 1.0:  # Way too high (pH > 8.0)
                reward -= 500.0  # EXTREME penalty - should never happen
            elif overshoot_amount > 0.5:  # Too high (pH > 7.5)
                reward -= 200.0  # MASSIVE penalty
            elif overshoot_amount > 0.2:  # Slightly high (pH > 7.2)
                reward -= 100.0  # Very heavy penalty
            elif overshoot_amount > 0.1:  # Just above (pH > 7.1)
                reward -= 50.0  # Heavy penalty
            else:  # pH 7.0-7.1 (acceptable range)
                reward -= 10.0  # Moderate penalty for being slightly above
        
        # 3. Progress bonus: Strongly reward getting closer (but penalize moving away)
        if self._last_dist is not None:
            progress = self._last_dist - dist  # Positive if getting closer
            if progress > 0:
                reward += 5.0 * progress  # Strong progress bonus
            else:
                # Moving away from target - penalize, especially if overshooting
                reward += 2.0 * progress  # Smaller penalty for moving away
                if pH > self.pH_target:
                    reward -= 20.0  # EXTRA HEAVY penalty for moving further above target
        self._last_dist = dist
        
        # 3.5. CRITICAL: Penalty for continuing after reaching target zone
        # If already in sweet spot (6.9-7.0), continuing is risky
        if 6.9 <= pH <= 7.0 and not terminated:
            # Small penalty for continuing when already in sweet spot
            # This encourages stopping early rather than risking overshoot
            reward -= 1.0
        
        # 4. pH zone rewards: Guide agent through titration stages
        # Buffer region (pH 3-6): Small positive reward for being in right range
        if 3.0 <= pH <= 6.0:
            reward += 1.0  # Small bonus for being in buffer region
        # Target zone (pH 6.5-7.0): Large reward (note: only up to 7.0, not above!)
        if 6.5 <= pH <= 7.0:
            reward += 5.0  # Big bonus for being near target
        # Sweet spot (pH 6.9-7.0): Huge reward (note: not above 7.0!)
        if 6.9 <= pH <= 7.0:
            reward += 20.0  # Massive bonus for being in sweet spot
        
        # 5. Volume-based guidance: Encourage reaching equivalence region
        V_ratio = self.Vb_L / self.Veq_L
        if 0.8 <= V_ratio <= 1.0:  # Approaching equivalence (but not past)
            reward += 2.0  # Bonus for being in equivalence region
        if 0.95 <= V_ratio <= 1.0:  # Very close to equivalence (but not past)
            reward += 5.0  # Bigger bonus
        # Penalty for going past equivalence when pH is already too high
        if V_ratio > 1.0 and pH > 7.0:
            reward -= 15.0  # Penalty for continuing past equivalence when already overshot
        
        # 6. Stopping rewards/penalties (only when agent chooses to stop)
        if terminated:
            # CRITICAL: Asymmetric stopping rewards - overshooting is MUCH worse
            if pH <= self.pH_target:  # Stopped at or below target (good!)
                if dist < 0.02:  # Extremely close
                    reward += 500.0  # EXTREME bonus for perfect hit
                elif dist < 0.05:
                    reward += 300.0  # MASSIVE bonus
                elif dist < 0.1:
                    reward += 150.0  # Very good bonus
                elif dist < 0.2:
                    reward += 50.0  # Good bonus
                elif dist < 0.5:
                    reward += 10.0  # Small bonus
            else:  # Stopped above target (OVERSHOT - BAD!)
                # EXTREME penalties that increase with overshoot
                overshoot = pH - self.pH_target
                if overshoot > 1.0:  # Way too high
                    reward -= 1000.0  # EXTREME penalty - should never happen
                elif overshoot > 0.5:
                    reward -= 500.0  # MASSIVE penalty
                elif overshoot > 0.2:
                    reward -= 200.0  # Very heavy penalty
                elif overshoot > 0.1:
                    reward -= 100.0  # Heavy penalty
                else:  # pH 7.0-7.1 (slight overshoot)
                    reward -= 30.0  # Moderate penalty
                
            # Volume-based stopping penalties
            if self.Vb_L < 0.01 * self.Veq_L:  # Stopped immediately
                reward -= 50.0  # Massive penalty
            elif self.Vb_L < 0.3 * self.Veq_L:  # Stopped way too early
                reward -= 15.0
            elif self.Vb_L < 0.7 * self.Veq_L:  # Stopped early
                reward -= 5.0
            elif self.Vb_L > 1.0 * self.Veq_L and pH > 7.0:  # Past equivalence AND overshot
                reward -= 100.0  # EXTREME penalty for overshooting past equivalence

        # truncate if max steps or extreme conditions
        # Allow more exploration - don't truncate too harshly
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            # Give reward based on final state even if truncated
            if dist < 0.1:
                reward += 10.0  # Good job even if ran out of steps
            elif dist < 0.5:
                reward += 2.0

        # Only terminate on truly extreme conditions
        if pH <= 0.0 or pH >= 14.0:
            terminated = True
            reward -= 30.0  # Heavy penalty for extreme pH
        
        # REALISTIC: Burette constraint already handled above
        # Allow some overshoot to learn from mistakes, but not excessive
        Vb_ml = self.Vb_L * 1000.0
        if Vb_ml > self.max_burette_ml * 1.1:  # Way beyond burette capacity
            terminated = True
            reward -= 25.0  # Penalty for trying to exceed physical limit

        obs = self._get_obs(pH)
        info = {"pH": pH, "Vb_ml": self.Vb_L * 1000.0, "dist": dist}

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Render the environment.
        
        Modes:
        - "human": Display using matplotlib (blocking)
        - "rgb_array": Return RGB array for video recording
        - "matplotlib": Return matplotlib figure (non-blocking)
        """
        if len(self.history) < 2:
            return None
            
        if mode == "human" or mode == "matplotlib":
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Extract trajectory data
            Vb_ml = [h[0] for h in self.history]
            pH_vals = [h[1] for h in self.history]
            colors = [h[2] for h in self.history]
            
            # Left plot: Titration curve
            ax1 = axes[0]
            ax1.plot(Vb_ml, pH_vals, 'o-', linewidth=2.5, markersize=8, 
                    color='steelblue', label='Titration curve', zorder=2)
            ax1.scatter([Vb_ml[0]], [pH_vals[0]], color='green', s=150, 
                       zorder=5, label='Start', marker='s')
            ax1.scatter([Vb_ml[-1]], [pH_vals[-1]], color='red', s=150, 
                       zorder=5, label='Current', marker='*')
            ax1.axhline(7.0, color='gray', linestyle='--', linewidth=2, 
                       alpha=0.7, label='Target pH 7.0', zorder=1)
            ax1.axvline(self.Veq_L * 1000, color='orange', linestyle=':', 
                       linewidth=2, alpha=0.7, label=f'Equivalence ({self.Veq_L*1000:.1f} mL)', zorder=1)
            ax1.set_xlabel('Base Volume Added (mL)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('pH', fontsize=12, fontweight='bold')
            ax1.set_title(f'Titration Progress (Step {self.step_count}/{self.max_steps})', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=10)
            ax1.set_ylim(0, 14)
            ax1.set_xlim(-1, max(Vb_ml) * 1.1 if Vb_ml else 60)
            
            # Right plot: Indicator color + info
            ax2 = axes[1]
            current_pH = pH_vals[-1]
            current_color = colors[-1]
            
            # Large indicator circle
            circle = Circle((0.5, 0.5), 0.35, transform=ax2.transAxes, 
                          facecolor=current_color, edgecolor='black', linewidth=3)
            ax2.add_patch(circle)
            
            # Info text
            info_text = f"""
Current State:
━━━━━━━━━━━━━━━━━━━━
pH: {current_pH:.2f}
Base Added: {Vb_ml[-1]:.2f} mL
V/Veq: {Vb_ml[-1] / (self.Veq_L * 1000):.2%}
Step: {self.step_count}/{self.max_steps}
Distance to Target: {abs(current_pH - self.pH_target):.2f}
            """.strip()
            
            ax2.text(0.5, 0.15, info_text, transform=ax2.transAxes,
                    fontsize=11, ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title('Indicator Color', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            if mode == "human":
                plt.show(block=False)
                plt.pause(0.01)  # Non-blocking pause
                return fig
            else:  # matplotlib mode
                return fig
                
        elif mode == "rgb_array":
            # Return RGB array for video recording
            fig = self.render(mode="matplotlib")
            if fig is None:
                return None
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return buf
        else:
            return None

