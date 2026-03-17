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
    findmaxvalue(s::State, V_itp::GFInterpolation, e::Environment)

Find the maximum possible value of the dynamic problem under state
`s` given (interpolated) value function `V_itp` and environment `e`. Return
a tuple `(maxval, action)` where `action` is the value-maximising action.
""" 
function findmaxvalue(s::State, V_itp::GFInterpolation, e::Environment)
    actions = actiongrid(e)

    maxval = typemin(eltype(GriddedFunctions.griddedfunction(V_itp)))
    maxI = CartesianIndex{ndims(actions)}()

    for I in eachindex(actions)
        val = actionvalue(actions[I], s, V_itp, e)
        if val >= maxval
            maxval = val
            maxI = I
        end
    end

    (maxval, actions[maxI])
end

"""
    bellman(V::AbstractGriddedFunction, e::Environment)

Compute the updated value function conditional on initial value function
(when solving a value function iteration problem) or next-period value
function (when solving an optimal path by backward induction from a terminal
value) `V`, and conditional on state of the environment `e`. Return a tuple
`(V_new, policy)` where `V_new` is the updated value function and `policy`
the corresponding policy function.

In its default implementation, this method efficiently iterates over the state
space (using parallel threads) and calls the [`findmaxvalue`](@ref) function
to determine the optimal value conditional on the current state.

If the structure of the dynamic problem allows for a more efficient approach,
or if heavy lifting using the GPU is required, this method can be overloaded
correspondingly.
"""
function bellman(V::AbstractGriddedFunction, e::Environment)

    V_itp = interpolate(V)
    states = grid(V)
    V_new = similar(V)
    policy = GriddedFunction(eltype(actiongrid(e)), states, undef)

    Threads.@threads for I in eachindex(V)
        V_new[I], policy[I] = findmaxvalue(states[I], V_itp, e)
    end

    (V_new, policy)
end
