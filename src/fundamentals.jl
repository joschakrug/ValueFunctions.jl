"""
    Variable

Abstract supertype for all model variables in an optimal control problem.
Both [`State`](@ref) and [`Action`](@ref) inherit from `Variable`.

Concrete subtypes support a NamedTuple-like interface:

- `length(v)` — number of fields
- `v[i::Int]` — field access by position (`v[begin]`, `v[end]` also work)
- `v[s::Symbol]` — field access by name
- `for x in v` — iterate over field values in declaration order
- `eltype(T)` — `promote_type` of all field types

See also: [`State`](@ref), [`Action`](@ref).
"""
abstract type Variable end

GriddedFunctions.dimnames(::Type{V}) where V <: Variable = fieldnames(V)

"""
    State

Abstract supertype for states of an optimal control problem.

Any state type to be passed to a value function solution algorithm must
inherit the abstract `State` type.

Any concrete subtype of `State` must implement the [`update`](@ref) and
[`currentvalue`](@ref) methods.

See also: [`Action`](@ref), [`StateGrid`](@ref), [`ActionGrid`](@ref).

# Example

```{julia}
struct FirmState <: State
    capital::Float64
end
```
"""
abstract type State <: Variable end

"""
    Action

Abstract supertype for actions in an optimal control problem.

Any action passed to a value function solution algorithm must
inherit the abstract `Action` type.

Any concrete subtype of `Action` must implement the [`update`](@ref) and
[`currentvalue`](@ref) methods.

See also: [`State`](@ref), [`StateGrid`](@ref), [`ActionGrid`](@ref).

# Example

```{julia}
struct FirmAction <: Action
    production::Float64
    investment::Float64
end
```
"""
abstract type Action <: Variable end

### implement full named-tuple like behaviour for Variable subtypes (State, Action)

Base.length(x::Variable)     = nfields(x)
Base.firstindex(x::Variable) = 1
Base.lastindex(x::Variable)  = nfields(x)

Base.getindex(x::Variable, i::Int)    = getfield(x, i)
Base.getindex(x::Variable, s::Symbol) = getfield(x, s)

function Base.iterate(x::Variable)
    nfields(x) == 0 && return nothing
    return (getfield(x, 1), 2)
end

function Base.iterate(x::Variable, i::Int)
    i > nfields(x) && return nothing
    return (getfield(x, i), i + 1)
end

function Base.eltype(::Type{T}) where {T <: Variable}
    ft = fieldtypes(T)
    isempty(ft) && return Union{}
    return promote_type(ft...)
end

Base.convert(::Type{T}, t::Tuple) where T <: Variable = T(t...)


"""
    Environment{P, AG <: ActionGrid, SG <: StateGrid}

Collects the complete specification of an optimal control problem: model
parameters, an action grid, and a state grid.

# Fields

- `params::P` — model parameters (of any type P)
- `actiongrid::AG` — grid of admissible actions (an [`ActionGrid`](@ref))
- `stategrid::SG` — grid of states (a [`StateGrid`](@ref))

See also: [`ActionGrid`](@ref), [`StateGrid`](@ref).
"""
struct Environment{P, AG <: AbstractGrid{<:Action}, SG <: AbstractGrid{<:State}}
    params::P
    actiongrid::AG
    stategrid::SG
end

actiongrid(e::Environment) = e.actiongrid
stategrid(e::Environment) = e.stategrid
parameters(e::Environment) = e.params

function Base.show(io::IO, e::Environment)
    print(io, "Environment(")
    show(io, e.params)
    print(io, ", ")
    show(io, e.actiongrid)
    print(io, ", ")
    show(io, e.stategrid)
    print(io, ")")
end
