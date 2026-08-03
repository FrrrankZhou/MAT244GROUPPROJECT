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

## Final-report rubric readiness

The checkboxes below distinguish implementation from a fully argued final
report. `Strong` means evidence already exists and mainly needs to be written
clearly. `Partial` means some implementation or evidence exists, but the report
still needs analysis. Unchecked items are not yet adequately completed.

### Project scope - 10 points

- [x] **Strong:** A substantial real-world ODE application is clearly defined.
- [x] **Strong:** The project goes beyond a one-direction textbook example by
  using synchronized two-component CESMD input and time-wise vector synthesis.
- [x] **Strong:** The primary goals are concrete: solve the response, compare
  numerical methods, calculate `r_max`, and study parameter sensitivity.
- [ ] **Needs strengthening:** State a focused research question near the start
  of the report, rather than presenting only a list of tasks.

### Mathematical and technical content - 50 points

- [x] **Strong:** The second-order base-excitation ODE and normalized equation
  are implemented correctly.
- [x] **Strong:** The conversion to a first-order system is explicit.
- [x] **Strong:** Euler and RK4 are implemented rather than delegated to a
  software ODE solver.
- [x] **Strong:** HNE and HNN are solved separately and combined at the same
  time index.
- [x] **Strong:** Parameter conversion among `m`, `Tn`, `zeta`, `d`, and `k` is
  implemented and tested.
- [x] **Strong:** Twenty unit tests cover parsing, parameter validation,
  numerical updates, 2D synthesis, peak extraction, and parameter grids.
- [ ] **Partial:** Euler and RK4 results are compared graphically and
  numerically, but the final report still needs an explanation of why Euler
  overpredicts the response.
- [ ] **Partial:** Exact-solution validation exists on the SHM branch, but it
  must be incorporated into the submitted report or merged into the final
  deliverable.
- [ ] **Needs strengthening:** Add a step-size or convergence study. One time
  step and one exact-solution plot do not fully establish numerical accuracy.
- [ ] **Partial:** Mass, damping, and stiffness sensitivity is computed, but
  each curve needs mathematical interpretation through `d/m`, `k/m`, natural
  frequency, and damping ratio.
- [ ] **Needs strengthening:** If resonance is claimed, support it with a
  natural-period sweep or frequency-content analysis. A local parameter peak
  alone is insufficient evidence.
- [ ] **Needs strengthening:** Compare at least one additional earthquake record
  or explain why the conclusions are intentionally limited to Calama.
- [ ] **Partial:** Assumptions and limitations are documented above, but they
  must also appear in the final report's discussion.
- [ ] **Needs strengthening:** Discuss implications critically: what the
  computed gap can suggest, what it cannot establish, and how model uncertainty
  affects the conclusion.

### References and background research - 10 points

- [ ] **Partial:** CESMD and the course topics have been identified as the data
  and mathematical background.
- [ ] **Needs strengthening:** Cite the CESMD record and CESMD format or data
  documentation formally.
- [ ] **Needs strengthening:** Cite the course textbook sections for the
  mass-spring-damper model, characteristic equation, Euler, and RK4.
- [ ] **Needs strengthening:** Add a structural-dynamics reference for the
  base-excitation equation, relative displacement, natural frequency, and
  damping ratio.
- [ ] **Needs strengthening:** Add an appropriate source for structural pounding
  or separation distance, and distinguish that literature from the simplified
  `2 r_max` assumption.
- [ ] **Needs strengthening:** Use one consistent citation style and provide a
  complete reference list.

### Organization and writing - 15 points

- [ ] **Partial:** Definitions, notation, units, and assumptions are organized
  in this README.
- [ ] **Needs strengthening:** Write the final report with a clear sequence:
  introduction, model derivation, methods, verification, earthquake results,
  parameter analysis, limitations, and conclusion.
- [ ] **Needs strengthening:** Introduce every symbol before using it and use
  consistent notation for damping (`d`, not alternating among `c` and `d`).
- [ ] **Needs strengthening:** Distinguish relative displacement, absolute
  displacement, ground displacement, and ground acceleration throughout.
- [ ] **Needs strengthening:** Edit for professional mathematical English and
  verify the course page-limit requirement before submission.

### Figures, examples, and simulations - 15 points

- [x] **Strong:** Ground acceleration, East/North displacement, and total
  horizontal displacement are shown with units.
- [x] **Strong:** The peak horizontal displacement is marked at its actual common
  East/North time index.
- [x] **Strong:** Euler and RK4 displacement and velocity comparisons are
  available.
- [x] **Strong:** The one-at-a-time parameter study labels RK4 as the method.
- [ ] **Partial:** Exact/Euler/RK4 SHM validation is available on the validation
  branch but still needs a report caption and interpretation.
- [ ] **Needs strengthening:** Give every final figure a numbered caption that
  states the dataset, parameters, method, units, and main conclusion.
- [ ] **Needs strengthening:** Refer to every figure and table in the report
  text; do not include figures without analysis.
- [ ] **Needs strengthening:** Add a compact table comparing records or time
  steps so the conclusions are supported by more than one visual example.


7. Verify the page limit and complete the final report structure.
