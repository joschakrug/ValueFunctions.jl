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
    optimalvalue!(objective::AbstractGriddedFunction, s::State, V_itp, e::Environment)

Return the value of the optimal control problem conditional on current state `s`,
initial (interpolated) value function `V_itp` and environment `e`. Store values
of the objective function generated in the process in `objective`.

In its default implementation, this method relies on a grid search over the
action grid in the provided [`Environment`](@ref), using the [`actionvalue`](@ref)
function to determine the value of each possible action. It can be overwritten
with a custom optimisation algorithm for user implementations of `Action` and
`State` if the structure of the dynamic control problem allows for it.
"""
function optimalvalue!(objective::AbstractGriddedFunction, s::State, V_itp::GFInterpolation, e::Environment)
    argmap!(a -> actionvalue(a, s, V_itp, e), objective)
    first(findmax(objective))
end

"""
    optimalvalue(s::State, V_itp::GFInterpolation, e::Environment)

Like [`optimalvalue!`](@ref) but allocates a new `objective::GriddedFunction` over the
action space under the hood. This does not matter for a single call but
drastically reduces performance under repeated calls.
""" 
function optimalvalue(s::State, V_itp::GFInterpolation, e::Environment)
    objective = GriddedFunction(Float64, actiongrid(e), undef)
    optimalvalue!(objective, s, V_itp, e)
end

"""
    bellman(V::AbstractGriddedFunction, e::Environment)

Return the updated value function conditional on initial value function
(when solving a value function iteration problem) or next-period value
function (when solving an optimal path by backward induction from a terminal
value) `V`, and conditional on state of the environment `e`.

In its default implementation, this method efficiently iterates over the state
space (using parallel threads) and calls the [`optimalvalue!`](@ref) function
to determine the optimal value conditional on the current state.

If the structure of the dynamic problem allows for a more efficient approach,
or if heavy lifting using the GPU is required, this method can be overloaded
correspondingly.
"""
function bellman(V::AbstractGriddedFunction, e::Environment)

    V_itp = interpolate(V)
    states = grid(V)
    V_new = similar(V)

    AG = typeof(actiongrid(e))
    OT = GriddedFunction{Float64, ndims(AG), AG}

    let pool = Channel{OT}(Threads.nthreads())
        foreach(_ -> put!(pool, GriddedFunction(Float64, actiongrid(e), undef)), 1:Threads.nthreads())

        Threads.@threads for I in eachindex(V)
            current_objective = take!(pool)
            V_new[I] = optimalvalue!(current_objective, states[I], V_itp, e)
            put!(pool, current_objective)
        end
    end

    V_new
end
