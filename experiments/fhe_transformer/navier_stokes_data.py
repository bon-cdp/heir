"""
2D Navier-Stokes Vorticity Data Generator

Pure spectral method for 2D incompressible Navier-Stokes in vorticity form.

Equations:
==========
∂ω/∂t + (v · ∇)ω = ν∇²ω    (vorticity transport)
∇²ψ = -ω                     (Poisson equation for stream function)
v = (∂ψ/∂y, -∂ψ/∂x)         (velocity from stream function)

where:
- ω(x,y,t) is vorticity
- ψ(x,y,t) is stream function
- v(x,y,t) is velocity field
- ν is kinematic viscosity

Why Vorticity Form?
==================
1. Natural for 2D: div(v) = 0 automatically satisfied
2. Single scalar equation (not vector Navier-Stokes)
3. Fourier modes are exact eigenfunctions of Laplacian
4. Connection to character theory: ω̂_k are Fourier coefficients!

Spectral Method:
===============
Represent ω in Fourier space:
    ω(x,y,t) = Σ_k ω̂_k(t) e^{ik·x}

Evolution equation becomes:
    dω̂_k/dt = -ν|k|² ω̂_k + N̂_k(t)

where N̂_k is nonlinear term (computed in physical space, transformed back).

This is PERFECT for wreath product attention:
- Each Fourier mode k is a character of the translation group!
- Position-dependent: different modes dominate at different (x,y)
- Multi-scale: small k (large eddies), large k (small eddies)

Initial Conditions:
==================
1. Taylor-Green Vortex (analytical):
   ω(x,y,0) = -2A sin(x) sin(y)

2. Random vorticity (turbulence):
   ω̂_k(0) ~ N(0, k^{-α}) for energy spectrum E(k) ~ k^α

3. Kolmogorov flow (forced turbulence):
   Add forcing f = F sin(4y) to maintain energy
"""

import numpy as np
from scipy.fftpack import fft2, ifft2, fftfreq
from typing import Tuple, List
import matplotlib.pyplot as plt


class NavierStokes2D:
    """
    2D incompressible Navier-Stokes in vorticity formulation.

    Uses pseudo-spectral method with 2/3 dealiasing rule.

    Example:
        ns = NavierStokes2D(N=128, L=2*np.pi, nu=0.001)
        omega0 = ns.taylor_green_vortex()
        omega_history = ns.simulate(omega0, dt=0.01, n_steps=1000)
    """

    def __init__(
        self,
        N: int = 128,
        L: float = 2 * np.pi,
        nu: float = 0.001,
        forcing: bool = False
    ):
        """
        Initialize 2D Navier-Stokes solver.

        Args:
            N: Grid resolution (N×N)
            L: Domain size [0, L] × [0, L] (periodic)
            nu: Kinematic viscosity (ν)
            forcing: Add Kolmogorov forcing for sustained turbulence
        """
        self.N = N
        self.L = L
        self.nu = nu
        self.forcing = forcing

        # Physical grid
        self.x = np.linspace(0, L, N, endpoint=False)
        self.y = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Wave numbers
        self.kx = fftfreq(N, L / (2 * np.pi * N))
        self.ky = fftfreq(N, L / (2 * np.pi * N))
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)

        # Squared wave number (for Laplacian)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1  # Avoid division by zero (will multiply by 0 anyway)

        # Dealiasing mask (2/3 rule)
        self.dealias_mask = (np.abs(self.KX) < N//3) & (np.abs(self.KY) < N//3)

        print(f"2D Navier-Stokes Solver:")
        print(f"  Grid: {N}×{N}")
        print(f"  Domain: [0, {L:.2f}] × [0, {L:.2f}]")
        print(f"  Viscosity: ν = {nu}")
        print(f"  Forcing: {forcing}")

    def taylor_green_vortex(self, A: float = 1.0) -> np.ndarray:
        """
        Taylor-Green vortex initial condition.

        Analytical solution with exponential decay in viscous case.

        Args:
            A: Amplitude

        Returns:
            Vorticity field ω(x,y,0)
        """
        omega = -2 * A * np.sin(self.X) * np.sin(self.Y)
        return omega

    def random_vorticity(
        self,
        k_max: int = None,
        alpha: float = 2.0,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Random initial vorticity with specified energy spectrum.

        E(k) ~ k^{-α} for turbulent initial conditions.

        Args:
            k_max: Maximum wavenumber (default: N//3)
            alpha: Spectral slope (α=2 for Kolmogorov)
            amplitude: Overall amplitude

        Returns:
            Random vorticity field
        """
        if k_max is None:
            k_max = self.N // 3

        # Create random Fourier coefficients
        omega_hat = np.zeros((self.N, self.N), dtype=complex)

        for i in range(self.N):
            for j in range(self.N):
                kx, ky = self.kx[i], self.ky[j]
                k = np.sqrt(kx**2 + ky**2)

                if k > 0 and k <= k_max:
                    # Energy spectrum E(k) ~ k^{-alpha}
                    # Amplitude of mode ~ k^{-alpha/2}
                    magnitude = amplitude * k**(-alpha/2)

                    # Random phase
                    phase = np.random.uniform(0, 2*np.pi)

                    omega_hat[i, j] = magnitude * np.exp(1j * phase)

        # Transform to physical space
        omega = np.real(ifft2(omega_hat))

        # Normalize
        omega = omega - omega.mean()

        return omega

    def vorticity_to_velocity(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity from vorticity via stream function.

        ∇²ψ = -ω  →  ψ̂_k = ω̂_k / |k|²
        v = (∂ψ/∂y, -∂ψ/∂x)

        Args:
            omega: Vorticity field

        Returns:
            (u, v) velocity components
        """
        # Transform to Fourier space
        omega_hat = fft2(omega)

        # Solve Poisson equation for stream function
        psi_hat = -omega_hat / self.K2
        psi_hat[0, 0] = 0  # Mean = 0

        # Compute velocity in Fourier space
        # u = ∂ψ/∂y = i*ky*ψ̂
        # v = -∂ψ/∂x = -i*kx*ψ̂
        u_hat = 1j * self.KY * psi_hat
        v_hat = -1j * self.KX * psi_hat

        # Transform back to physical space
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))

        return u, v

    def rhs(self, omega: np.ndarray) -> np.ndarray:
        """
        Right-hand side of vorticity equation.

        dω/dt = -v·∇ω + ν∇²ω + f

        Args:
            omega: Vorticity field

        Returns:
            Time derivative dω/dt
        """
        # Transform to Fourier space
        omega_hat = fft2(omega)

        # Diffusion: ν∇²ω = -ν|k|² ω̂
        diffusion_hat = -self.nu * self.K2 * omega_hat

        # Nonlinear term: -(v·∇ω)
        # Compute in physical space to handle nonlinearity
        u, v = self.vorticity_to_velocity(omega)

        # Gradients of ω
        domega_dx_hat = 1j * self.KX * omega_hat
        domega_dy_hat = 1j * self.KY * omega_hat

        domega_dx = np.real(ifft2(domega_dx_hat))
        domega_dy = np.real(ifft2(domega_dy_hat))

        # Advection term
        advection = -(u * domega_dx + v * domega_dy)
        advection_hat = fft2(advection)

        # Apply dealiasing
        advection_hat *= self.dealias_mask

        # Total RHS
        rhs_hat = diffusion_hat + advection_hat

        # Add forcing if requested
        if self.forcing:
            # Kolmogorov forcing: f = F sin(4y)
            force = 0.1 * np.sin(4 * self.Y)
            force_hat = fft2(force)
            rhs_hat += force_hat

        # Transform back
        return np.real(ifft2(rhs_hat))

    def step_rk4(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Single RK4 timestep.

        Args:
            omega: Current vorticity
            dt: Time step

        Returns:
            Updated vorticity
        """
        k1 = dt * self.rhs(omega)
        k2 = dt * self.rhs(omega + 0.5 * k1)
        k3 = dt * self.rhs(omega + 0.5 * k2)
        k4 = dt * self.rhs(omega + k3)

        omega_new = omega + (k1 + 2*k2 + 2*k3 + k4) / 6

        return omega_new

    def simulate(
        self,
        omega0: np.ndarray,
        dt: float,
        n_steps: int,
        save_every: int = 1
    ) -> List[np.ndarray]:
        """
        Simulate vorticity evolution.

        Args:
            omega0: Initial vorticity
            dt: Time step
            n_steps: Number of steps
            save_every: Save frequency

        Returns:
            List of vorticity snapshots
        """
        history = [omega0.copy()]
        omega = omega0.copy()

        for step in range(n_steps):
            omega = self.step_rk4(omega, dt)

            if (step + 1) % save_every == 0:
                history.append(omega.copy())

                if (step + 1) % 100 == 0:
                    print(f"  Step {step+1}/{n_steps}")

        return history

    def energy_spectrum(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute energy spectrum E(k).

        Args:
            omega: Vorticity field

        Returns:
            (k_bins, E_k) energy spectrum
        """
        omega_hat = fft2(omega)

        # Bin by wavenumber magnitude
        k_mag = np.sqrt(self.K2)
        k_max = self.N // 3
        k_bins = np.arange(0, k_max)
        E_k = np.zeros(len(k_bins))

        for i_bin, k in enumerate(k_bins):
            mask = (k_mag >= k) & (k_mag < k + 1)
            E_k[i_bin] = np.sum(np.abs(omega_hat[mask])**2)

        return k_bins, E_k


def generate_training_data(
    N: int = 64,
    nu: float = 0.001,
    dt: float = 0.01,
    n_steps: int = 500,
    save_every: int = 5
) -> Tuple[List[np.ndarray], NavierStokes2D]:
    """
    Generate training data for wreath product attention.

    Args:
        N: Grid size
        nu: Viscosity
        dt: Time step
        n_steps: Number of steps
        save_every: Save frequency

    Returns:
        (vorticity_snapshots, solver)
    """
    print("\n" + "="*70)
    print("Generating 2D Navier-Stokes Training Data")
    print("="*70)

    solver = NavierStokes2D(N=N, L=2*np.pi, nu=nu)

    # Taylor-Green initial condition
    print("\nInitial condition: Taylor-Green vortex")
    omega0 = solver.taylor_green_vortex()

    print(f"\nSimulating {n_steps} steps (dt={dt})...")
    history = solver.simulate(omega0, dt, n_steps, save_every)

    print(f"\n✅ Generated {len(history)} vorticity snapshots")
    print(f"   Shape: {history[0].shape}")
    print(f"   Time span: 0 to {n_steps * dt:.2f}")

    return history, solver


if __name__ == "__main__":
    # Generate data
    history, solver = generate_training_data(
        N=64,
        nu=0.001,
        dt=0.01,
        n_steps=200,
        save_every=2
    )

    # Show energy spectrum evolution
    print("\nEnergy spectrum at different times:")
    for i, omega in enumerate(history[::25]):
        k_bins, E_k = solver.energy_spectrum(omega)
        E_total = np.sum(E_k)
        print(f"  t={i*25*2*0.01:.2f}: E_total = {E_total:.6f}")

    print("\n" + "="*70)
    print("✅ Navier-Stokes data generation complete!")
    print("="*70)
    print("\nReady for wreath product attention experiments.")
