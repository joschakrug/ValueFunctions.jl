# A toolbox for value function iteration in Julia

This package provides tools to comfortably and efficiently run value function iteration algorithms in Julia. It relies heavily on the [GriddedFunctions.jl](https://github.com/joschakrug/GriddedFunctions.jl) package but adds some convenience layers specific to value function iteration on top.

## Usage

## Developing

**Note:** This package imports the GriddedFunctions.jl package as a submodule in the `GriddedFunctions` directory. If a feature of this package requires a change to the GriddedFunctions.jl package, the envisioned workflow is to

- update the original GriddedFunctions.jl package and push any changes made
- check out the new version of the GriddedFunctions.jl package from GitHub into this repository.

See the [git-scm.com handbook section on submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for more details.
