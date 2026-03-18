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
function ValueFunctions.nextstate(a::FirmAction, s::FirmState, ::Environment)
    FirmState(0.9 * s.k + a.i)
end

# Period payoff: CES revenue R(q) = A·q^((σ−1)/σ) minus investment cost
# Returns -Inf for infeasible actions (output exceeds capital; negative investment)
function ValueFunctions.actionvalue(a::FirmAction, s::FirmState, V_itp, e::Environment)
    σ, A, β = e.params.σ, e.params.A, e.params.β
    q, i = a.q, a.i

    (0.0 < q ≤ s.k && i ≥ 0.0) || return -Inf

    revenue = A * q^((σ - 1) / σ)
    revenue - i + β * V_itp(nextstate(a, s, e))
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

## Installation

Installation

To install the latest stable version, add [JuliaRegistryJKG](https://github.com/joschakrug/JuliaRegistryJKG) to your Julia registries by running

```julia
Pkg.Registry.add(RegistrySpec(url="git@github.com:joschakrug/JuliaRegistryJKG.git"))
```

in your REPL. With this registry added, you can simply `] add` and `] update` the ValueFunctions package using your package manager.

To install the latest development version, clone this git repository to a local folder and add that folder to your main project as a development dependency running `] dev local/repo/path`. Bear in mind that this package depends on [GriddedFunctions.jl](https://github.com/joschakrug/GriddedFunctions.jl) which is not yet available in Julia's General registry. That means you will have to add `JuliaRegistryJKG` in any case, even if you just want to use the development version.

## For developers

### Testing

Testing is as simple as running `] test ValueFunctions` with the `test` environment activated. Manual tests in the `test` folder require the `test` environment to be activated as well.

### Pushing updated versions

To register an updated package version in `JuliaRegistryJKG`, bump the version number in the `Project.toml`, push a tagged commit with the same version number to GitHub and then run

```julia
julia> using LocalRegistry
julia> register("/path/to/local/copy/of/project", registry = "JuliaRegistryJKG", push = true)
```
