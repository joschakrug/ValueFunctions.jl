# A toolbox for value function iteration in Julia

This package provides tools to comfortably and efficiently run value function iteration algorithms in Julia. It relies heavily on the [GriddedFunctions.jl](https://github.com/joschakrug/GriddedFunctions.jl) package but adds specific value function iteration functionality on top.

## Usage

Define `State` and `Action` types for your problem, construct grids and an `Environment`:

```julia
using ValueFunctions
using GriddedFunctions

struct FirmState <: State
    k::Float64  # capital stock
end

struct FirmAction <: Action
    q::Float64  # output quantity
    i::Float64  # investment
end

struct Parameters
    σ::Float64  # elasticity of demand
    A::Float64  # demand shifter
    β::Float64  # discount factor
end

actiongrid = Grid(
    FirmAction,
    q = LinearAxis(range(0.0, 1.0; length = 100)),
    i = LinearAxis(range(0.0, 1.0; length = 100))
)

stategrid = Grid(FirmState, LinearAxis(range(0.0, 5.0; length = 500)))

env = Environment(Parameters(1.5, 1.0, 0.9), actiongrid, stategrid)
```

Implement `nextstate` and `actionvalue` for your model, then run value function iteration
by iterating the Bellman operator until convergence:

```julia
# Capital dynamics: k' = (1 − δ)·k + i, with 10% depreciation
function ValueFunctions.nextstate(s::FirmState, a::FirmAction, ::Environment)
    FirmState(0.9 * s.k + a.i)
end

# Period payoff: CES revenue R(q) = A·q^((σ−1)/σ) minus investment cost
# Returns -Inf for infeasible actions (output exceeds capital; negative investment)
function ValueFunctions.actionvalue(a::FirmAction, s::FirmState, V_itp, e::Environment)
    σ, A, β = e.params.σ, e.params.A, e.params.β
    q, i = a.q, a.i

    (0.0 < q ≤ s.k && i ≥ 0.0) || return -Inf

    revenue = A * q^((σ - 1) / σ)
    revenue - i + β * V_itp(nextstate(s, a, e))
end

# Zero initial value function
V = GriddedFunction(Float64, stategrid, _ -> 0.0)

# Iterate the Bellman operator until convergence
for _ in 1:2000
    V_new = bellman(V, env)
    maximum(abs(v_new - v) for (v_new, v) in zip(V_new, V)) < 1e-6 && break
    V = V_new
end
```

## Developing

**Note:** This package imports the GriddedFunctions.jl package as a submodule in the `deps/GriddedFunctions` directory. If a feature of this package requires a change to the GriddedFunctions.jl package, the envisioned workflow is to

- update the original GriddedFunctions.jl package and push any changes made
- check out the new version of the GriddedFunctions.jl package from GitHub into this repository.

See the [git-scm.com handbook section on submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for more details.
