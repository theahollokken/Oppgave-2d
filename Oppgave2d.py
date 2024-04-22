#kode til sammenlikning av 2c og 2d

import numpy as np
import matplotlib.pyplot as plt

# Definer parametere
L = 10.0  # lengden på romdomenet
T = 2.0  # total tid
m = 100  # antall romlige punkter
n = 100  # antall tidssteg
dx = L / (m - 1)  # romlig steglengde
dt = T / (n - 1)  # tidssteglengde
v = 0.1  # viskositet for 2c

# Initialiser domenet
x = np.linspace(0, L, m)
t = np.linspace(0, T, n)

# Initialbetingelser
def initial_conditions(x):
    return np.sin(np.pi * x / L)

# Initialiser løsningsmatrisene
u_visc = np.zeros((n, m))
u_invisc = np.zeros((n, m))
u_visc[0, :] = initial_conditions(x)
u_invisc[0, :] = initial_conditions(x)

# FTCS metode for Burgers' ligning med viskositet (Oppgave 2c)
for i in range(1, n):
    for j in range(1, m-1):
        u_x = (u_visc[i-1, j+1] - u_visc[i-1, j-1]) / (2 * dx)
        u_xx = (u_visc[i-1, j+1] - 2*u_visc[i-1, j] + u_visc[i-1, j-1]) / (dx**2)
        u_visc[i, j] = u_visc[i-1, j] - dt * u_visc[i-1, j] * u_x + v * dt * u_xx

# Lax-Friedrichs metode for ikke-viskøs Burgers' ligning (Oppgave 2d)
for i in range(1, n):
    for j in range(1, m-1):
        u_invisc[i, j] = 0.5 * (u_invisc[i-1, j+1] + u_invisc[i-1, j-1]) \
                       - dt / (2 * dx) * (u_invisc[i-1, j+1]**2 - u_invisc[i-1, j-1]**2) / 2

# Beregn maksimale absolutte residualer for begge metoder
res_visc = np.max(np.abs(u_visc[-1, :] - initial_conditions(x)))
res_invisc = np.max(np.abs(u_invisc[-1, :] - initial_conditions(x)))

print(f'Maksimalt residual for viskøs løsning: {res_visc}')
print(f'Maksimalt residual for ikke-viskøs løsning: {res_invisc}')

# Plot løsningene for visualisering
plt.figure(figsize=(10, 5))
plt.plot(x, u_visc[-1, :], label='Viskøs (FTCS)')
plt.plot(x, u_invisc[-1, :], label='Ikke-viskøs (Lax-Friedrichs)', linestyle='--')
plt.legend()
plt.title('Sammenligning av viskøs og ikke-viskøs løsning ved t=T')
plt.xlabel('Posisjon (x)')
plt.ylabel('Hastighet (u)')
plt.show()