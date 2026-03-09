"""
    nextstate(a::Action, s::State, e::Environment)

Return the next-period state given current state `s`, action `a` and environment
`e`.

This method must be implemented for any user implementations of `State` and
`Action` for the [`bellman`](@ref) equation to work.
"""
function nextstate end

"""
    actionvalue(a::Action, s::State, V_itp::GFInterpolation, e::Environment)

Return the value of action `a` in the optimal control problem conditional on
current state `s`, initial (interpolated) value function `V_itp` and environment
`e`.

This method must be implemented for any user implementations of `State` and
`Action` for the [`bellman`](@ref) equation to work.
"""
function actionvalue end

"""
    value!(objective::GriddedVariableFunction{Action}, s::State, V_itp, e::Environment)

Return the value of the optimal control problem conditional on current state `s`,
initial (interpolated) value function `V_itp` and environment `e`. Store values
of the objective function generated in the process in `objective`.

In its default implementation, this method relies on a grid search over the
action grid in the provided [`Environment`](@ref), using the [`actionvalue`](@ref)
function to determine the value of each possible action. It can be overwritten
with a custom optimisation algorithm for user implementations of `Action` and
`State` if the structure of the dynamic control problem allows for it.
"""
function value!(objective::AbstractGriddedFunction, s::State, V_itp::GFInterpolation, e::Environment)
    argmap!(a -> actionvalue(a, s, V_itp, e), objective)
    first(findmax(objective))
end

"""
    value(s::State, V_itp::GFInterpolation, e::Environment)

Like [`value!`](@ref) but allocates a new `objective::GriddedFunction` over the
action space under the hood. This does not matter for a single call but
drastically reduces performance under repeated calls.
""" 
function value(s::State, V_itp::GFInterpolation, e::Environment)
    objective = GriddedFunction(Float64, actiongrid(e), undef)
    value!(objective, s, V_itp, e)
end

"""
    bellman(V::GriddedVariableFunction{S}, e::Environment) where S <: State

Return the updated value function conditional on initial value function
(when solving a value function iteration problem) or next-period value
function (when solving an optimal path by backward induction from a terminal
value) `V`, and conditional on state of the environment `e`.

In its default implementation, this method uses a simple grid search algorithm
across all values on the action grid specified in `e`. It requires that the
[`value`](@ref) and [`nextstate`](@ref) methods are implemented for the
respective user implementations of `Action` and `State`.

If the structure of the problem allows for a more efficient solution algorithm
than grid search, this method can be overridden.
"""
function bellman(V::AbstractGriddedFunction, e::Environment)

    V_itp = interpolate(V)

    # initialise undefined objective function over the action grid
    # (this is just to allocate the array, it will be re-filled at every
    # iteration over each point on the state grid)
    objective = GriddedFunction(Float64, actiongrid(e), undef)
    argmap(state -> value!(objective, state, V_itp, e), V)
end
