"""
Implements a standardised interface for value function iteration that makes it
easy to solve optimal control problems that can be adequately represented in
value function form.

To use it, define State and Action structs based on your own problem.

# Example

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
end

actiongrid = Grid(
    FirmAction, 
    (
        q = LinearAxis(0, 1, length = 100),
        i = LinearAxis(0, 1, length = 100)
    )
)

stategrid = Grid(
    FirmState,
    (k = LinearAxis(0, 5, length = 500),)
)

env = Environment(
    Parameters(1.5, 1), actiongrid, stategrid
)
```
"""
module ValueFunctions

using GriddedFunctions

include("fundamentals.jl")
include("vfi.jl")

export Action, State, Environment
export stategrid, actiongrid
export nextstate, actionvalue, optimalvalue, optimalvalue!, bellman

end