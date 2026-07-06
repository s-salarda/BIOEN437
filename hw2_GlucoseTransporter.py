import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Write and plot a simple glucose transporter
def glucose_transport_odes(y, t, k1, k_1, k2, k3, k_3):
    """
    ODE system for a simple glucose transporter
    
    Variables:
    Se = glucose concentration outside cell
    Ce = free carrier protein concentration outside
    Pe = glucose-carrier complex concentration outside
    Si = glucose concentration inside cell
    Ci = free carrier protein concentration inside
    Pi = glucose-carrier complex concentration inside
    
    Reactions:
    Outside: Se + Ce ⇌ Pe (rates k1, k_1)
    Conformational Flip: Ce → Ci (k2)
    Conformational Flip: Pe → Pi (k2)
    Inside: Si+ Ci ⇌ Pi (rates k3, k_3)

    """

    # State Variables
    Se, Ce, Pe, Si, Ci, Pi = y
    
    # Outside bindng and transport: Se + Ce ⇌ Pe
    dSe_dt = -k1 * Se * Ce + k_1 * Pe
    dCe_dt = k2 * Ci - k2 * Ce + k_1 * Pe - k1 * Se * Ce
    dPe_dt = k2 * Pi - k2 * Pe + k1 * Se * Ce - k_1 * Pe
    
    # Inside binding and transport: Si + Ci ⇌ Pi
    dSi_dt = - k3 * Si * Ci + k_3 * Pi
    dCi_dt = k2 * Ce - k2 * Ci + k_3 * Pi - k3 * Si * Ci
    dPi_dt = k2 * Pe - k2 * Pi + k3 * Si * Ci - k_3 * Pi
    
    return [dSe_dt, dCe_dt, dPe_dt, dSi_dt, dCi_dt, dPi_dt]

# Rate Constants 
k1 = 0.1    # forward binding rate outside (1/mM·s)
k_1 = 0.3  # reverse binding rate outside (1/s)
k2 = 0.02   # transport rate out→in (1/s)
k3 = 0.1   # forward binding rate inside (1/mM·s)
k_3 = 0.03  # reverse binding rate inside (1/s)

# Initial conditions
Se_0 = 5.0   # initial glucose outside (mM)
Ce_0 = 1.0    # initial free carrier outside (mM)
Pe_0 = 0.0   # initial complex outside (mM)

Si_0 = 0.5     # initial glucose inside (mM)
Ci_0 = 0.0     # initial free carrier inside (mM)
Pi_0 = 0.0    # initial complex inside (mM)

y0_bidirectional = [Se_0, Ce_0, Pe_0, Si_0, Ci_0, Pi_0]

# Time points
t_long = np.linspace(0, 200, 2000)

# Solve ODE system
solution = odeint(glucose_transport_odes, y0_bidirectional, t_long, 
                                args=(k1, k_1, k2, k3, k_3))

Se, Ce, Pe, Si, Ci, Pi = solution.T

# Create comprehensive plot
fig, ((ax1), (ax2), (ax3), (ax4)) = plt.subplots(4, 1, figsize=(12,16))

# Plot 1: Glucose concentrations
ax1.plot(t_long, Se, 'b-', linewidth=2, label='Glucose Outside')
ax1.plot(t_long, Si, 'r-', linewidth=2, label='Glucose Inside')
ax1.plot(t_long, Se + Si, 'k--', linewidth=1, alpha=0.7, label='Total Glucose')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Glucose Concentration (mM)')
ax1.set_title('Glucose Distribution Across Cell Membrane')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Carrier proteins
ax2.plot(t_long, Ce, 'b-', linewidth=2, label='Free Carrier Outside')
ax2.plot(t_long, Ci, 'r-', linewidth=2, label='Free Carrier Inside')
ax2.plot(t_long, Pe, 'g-', linewidth=2, label='Complex Outside')
ax2.plot(t_long, Pi, 'm-', linewidth=2, label='Complex Inside')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Carrier Concentration (mM)')
ax2.set_title('Carrier Protein Dynamics')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Transport rates
transport_rate_in = k2 * Pe
transport_rate_out = k2 * Pi
net_transport_rate = transport_rate_in - transport_rate_out
ax3.plot(t_long, transport_rate_in, 'b-', linewidth=2, label='Rate In (out→in)')
ax3.plot(t_long, transport_rate_out, 'r-', linewidth=2, label='Rate Out (in→out)')
ax3.plot(t_long, net_transport_rate, 'k-', linewidth=2, label='Net Rate')
ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Transport Rate (mM/s)')
ax3.set_title('Transport Rates')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Concentration ratio and equilibrium approach
concentration_ratio = Si / (Se + 1e-10)  # Avoid division by zero
ax4.plot(t_long, concentration_ratio, 'purple', linewidth=2, label='[G_in]/[G_out]')
ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equilibrium (ratio=1)')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Concentration Ratio')
ax4.set_title('Approach to Equilibrium')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('glucose.png', dpi=300)
plt.show()

# Print system analysis
print("Glucose Transport Analysis:")
print("=" * 50)
print(f"Initial conditions:")
print(f"  Outside: [G] = {Se_0:.1f} mM, [C] = {Ce_0:.1f} mM")
print(f"  Inside:  [G] = {Si_0:.1f} mM, [C] = {Ci_0:.1f} mM")
print(f"\nRate constants:")
print(f"  Outside binding: k₁ = {k1:.3f} 1/(mM·s), k₋₁ = {k_1:.3f} 1/s")
print(f"  Inside binding:  k₃ = {k3:.3f} 1/(mM·s), k₋₃ = {k_3:.3f} 1/s")
print(f"  Transport rates: k₂ = {k2:.3f} 1/s (out→in), k₄ = {k2:.3f} 1/s (in→out)")
print(f"\nFinal steady-state values (t = {t_long[-1]:.0f} s):")
print(f"  Outside: [G] = {Se[-1]:.3f} mM, [C] = {Ce[-1]:.3f} mM, [GC] = {Pe[-1]:.3f} mM")
print(f"  Inside:  [G] = {Si[-1]:.3f} mM, [C] = {Ci[-1]:.3f} mM, [GC] = {Pi[-1]:.3f} mM")
print(f"  Concentration ratio [G_in]/[G_out] = {concentration_ratio[-1]:.3f}")
print(f"  Net transport rate = {net_transport_rate[-1]:.6f} mM/s")

# Conservation checks
total_glucose = Se + Si + Pe + Pi 
total_carrier = Ci + Ce + Pe + Pi

print(f"\nConservation checks:")
print(f"  Total glucose: initial = {Se_0:.1f}, final = {total_glucose[-1]:.3f} mM")
print(f"  Total carrier: initial = {Ce_0:.1f}, final = {total_carrier[-1]:.3f} mM")

# Conservation Test
assert np.allclose(total_glucose, total_glucose[0], rtol=1e-6, atol=1e-8)
assert np.allclose(total_carrier, total_carrier[0], rtol=1e-6, atol=1e-8)

# Behavioral Analysis
# Change Extracellular Glucose
t = np.linspace(0, 100, 2000)
plt.figure(figsize=(10,6))

for Se0 in np.arange(1.0, 6.5, 0.5):
    y0 = [Se0, Si_0, Ce_0, Ci_0, Pe_0, Pi_0]
    solution2 = odeint(glucose_transport_odes, y0, t, 
                                args=(k1, k_1, k2, k3, k_3))
    Se, Ce, Pe, Si, Ci, Pi = solution2.T
    plt.plot(t, Si, label=f'Se0={Se0:.1f} mM')

plt.xlabel('Time (s)')
plt.ylabel('Intracellular Glucose (mM)')
plt.title('Effect of Extracellular Glucose')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Changing K values but maintaining a Km=3mM
kvalues = [0.05, 0.1, 0.2, 0.4]
k_values = [3/0.05, 3/0.1, 3/0.2, 3/0.4]

# Increasing k1=k3 values
for k, k_ in zip(kvalues, k_values):
    solution4 = odeint(glucose_transport_odes, y0_bidirectional, t, 
                        args=(k, k_, k2, k, k_))
    Se, Ce, Pe, Si, Ci, Pi = solution4.T
    plt.plot(t, Si, label=f'Si (k1=k3={k:.2f})')
plt.title('Impact of k1 = k3 on Intracellular Glucose')
plt.xlabel('Time (s)')
plt.ylabel('[Glucose Inside] (mM)')
plt.grid(True, alpha=0.3)
plt.legend(fontsize='small', ncol=2, loc='lower right')
plt.tight_layout()
plt.show()

for k, k_ in zip(kvalues, k_values):
    solution4 = odeint(glucose_transport_odes, y0_bidirectional, t, 
                    args=(k, k_, k2, k, k_))
    Se, Ce, Pe, Si, Ci, Pi = solution4.T
    plt.plot(t, Se, label=f'Se (k1=k3={k:.2f})')
plt.title('Impact of k1 = k3 on Extracellular Glucose')
plt.xlabel('Time (s)')
plt.ylabel('[Glucose Inside] (mM)')
plt.grid(True, alpha=0.3)
plt.legend(fontsize='small', ncol=2, loc='lower right')
plt.tight_layout()
plt.show()

# Increaseing k_1=k_3 values
for k, k_ in zip(kvalues, k_values):
    solution5 = odeint(glucose_transport_odes, y0_bidirectional, t, 
                    args=(k_, k, k2, k_, k))
    Se, Ce, Pe, Si, Ci, Pi = solution5.T
    plt.plot(t, Si, label=f'Si (k_1=k_3={k:.2f})')
plt.title('Impact of k_1 = k_3 on Intracellular Glucose')
plt.xlabel('Time (s)')
plt.ylabel('[Glucose Inside] (mM)')
plt.grid(True, alpha=0.3)
plt.legend(fontsize='small', ncol=2, loc='lower right')
plt.tight_layout()
plt.show()

for k, k_ in zip(kvalues, k_values):
    solution5 = odeint(glucose_transport_odes, y0_bidirectional, t, 
                    args=(k_, k, k2, k_, k))
    Se, Ce, Pe, Si, Ci, Pi = solution5.T
    plt.plot(t, Se, label=f'Se (k_1=k_3={k:.2f})')
plt.title('Impact of k_1 = k_3 on Extracellular Glucose')
plt.xlabel('Time (s)')
plt.ylabel('[Glucose Inside] (mM)')
plt.grid(True, alpha=0.3)
plt.legend(fontsize='small', ncol=2, loc='lower right')
plt.tight_layout()
plt.show()
