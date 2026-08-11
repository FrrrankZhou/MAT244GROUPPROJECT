# Two-Dimensional Building Response to Earthquake Ground Motion

This project models a building as a linear single-degree-of-freedom (SDOF)
mass-spring-damper system. It uses the two orthogonal horizontal components of
a CESMD strong-motion record, HNE and HNN, to calculate the building's relative
East and North displacements during an earthquake.

The main course objectives are to formulate a second-order forced ODE, rewrite
it as a first-order system, implement Euler's method and RK4, and interpret the
resulting displacement rather than treating the numerical solver as a black
box.

## Mathematical model

Let `u_g(t)` be the ground displacement, `u(t)` the absolute displacement of
the building mass, and

```text
x(t) = u(t) - u_g(t)
```

the displacement of the building relative to the moving ground. In either
horizontal direction, the base-excited SDOF equation is

```text
m x'' + d x' + k x = -m a_g(t),
```

where `a_g(t) = u_g''(t)` is the measured ground acceleration. Dividing by
mass gives the equation actually evaluated by the program:

```text
x'' + (d/m) x' + (k/m) x = -a_g(t).
```

With velocity `v = x'`, this becomes the first-order system

```text
x' = v,
v' = -(d/m)v - (k/m)x - a_g(t).
```

This system is solved independently for HNE and HNN. The two solutions are
combined at every common time sample:

```text
r(t) = sqrt(x_E(t)^2 + x_N(t)^2).
```

Therefore, the reported maximum is

```text
r_max = max_t r(t),
```

not the larger of the two component maxima and not a combination of component
maxima occurring at different times.

The simplified separation estimate used in this project is

```text
G = 2 r_max.
```

## Assumptions and limitations

### Physical assumptions

- The building is represented by one lumped mass in each horizontal direction.
- The model is linear elastic: `m`, `d`, and `k` are constant during the event.
- HNE and HNN use the same mass, damping, and stiffness. The building is
  therefore treated as horizontally isotropic.
- East and North responses are uncoupled. Torsion and cross-direction coupling
  are neglected.
- The output `x(t)` is relative displacement with respect to the moving ground,
  not absolute displacement in an inertial coordinate system.
- Initial relative displacement and velocity are zero for the earthquake runs.
- Vertical motion, permanent deformation, yielding, structural damage,
  soil-structure interaction, and interaction between adjacent buildings are
  outside the model.

### Ground-motion assumptions

- HNE and HNN are treated as synchronized orthogonal components from the same
  CESMD station record.
- CESMD corrected acceleration (`.acc.V2c`) is used. Values in `cm/s^2` are
  converted to `m/s^2`.
- Sampling is uniform, and both horizontal components must have the same number
  of samples and sampling interval.
- No additional filtering, baseline correction, or rotation of the horizontal
  axes is performed after reading the corrected CESMD record.

### Numerical assumptions

- Explicit Euler and classical fourth-order Runge-Kutta are implemented
  directly for the first-order state system.
- RK4 uses linear interpolation of ground acceleration at the half steps.
- The time step is the sampling interval of the ground-motion record.
- Explicit Euler can accumulate artificial energy or become unstable in an
  oscillatory system. Its result is included as a numerical-method comparison,
  not treated as the most reliable prediction.
- The RK4 result is used for the parameter study.

### Interpretation of the parameter study

- Mass, damping, and stiffness are varied one at a time. The other two physical
  parameters remain fixed.
- Changing mass alone changes both `d/m` and `k/m`. Changing stiffness changes
  the natural frequency, and changing damping changes the damping ratio.
- A feature in an `r_max` curve cannot automatically be called resonance
  without relating the building frequency to the frequency content of the
  earthquake record.

### Limitation of the gap estimate

`G = 2 r_max` is only a screening-level simplification. It assumes two similar
buildings could reach equal displacements toward one another. Real separation
design depends on two different structural responses, their relative phase,
inelastic behavior, uncertainty, safety factors, and applicable building codes.
The value produced here is not a code-compliant design recommendation.

## Parameters

The example in `main.py` uses

```text
mass m            = 1,000,000 kg
natural period Tn = 1.0 s
damping ratio zeta = 0.05
```

and converts them using

```text
omega_n = 2 pi / Tn,
k = m omega_n^2,
d = 2 zeta m omega_n.
```

`BuildingParameters` also accepts directly specified `mass`, `stiffness`, and
`damping` values.

## Project files

```text
data_reader.py    CESMD .acc.V2c, ZIP, and nested-ZIP reader
ode_solver.py     model parameters, Euler, RK4, 2D synthesis, and peak analysis
plotting.py       response, method-comparison, and parameter-study figures
main.py           complete Calama experiment and output generation
test_core.py      unittest suite for parsing, solving, and analysis
results/          report-ready figures and numerical tables
```

The separate branch `codex/simple-harmonic-validation` contains a two-direction
simple harmonic motion experiment comparing the exact solution, Euler, and RK4.

## Running the project

The code requires Python, NumPy, and Matplotlib. From the project directory:

```powershell
.\.venv\Scripts\python.exe main.py
```

To run the tests without installing pytest:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_core
```

The main program reads `Calama.zip` and writes the following report artifacts:

```text
results/earthquake_response.png
results/euler_vs_rk4_displacement.png
results/euler_vs_rk4_velocity.png
results/parameter_study.png
results/peak_displacements.csv
```

For the current Calama run:

| Method | Maximum horizontal displacement | Peak time | Simplified gap |
|---|---:|---:|---:|
| Euler | 0.074240 m | 53.380 s | 0.148479 m |
| RK4 | 0.052046 m | 52.850 s | 0.104091 m |

The sizeable Euler-RK4 difference is itself an important numerical result and
should be discussed together with step-size sensitivity and the known behavior
of explicit Euler on oscillatory problems.


7. Verify the page limit and complete the final report structure.
