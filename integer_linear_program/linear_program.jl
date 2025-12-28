#==========================================================================
Implementation of a integer linear program for 2-Strong Roman Domination
Author: Atílio Gomes Luiz
Data: December, 27th 2025
===========================================================================#
# Importing the necessary packages
using DelimitedFiles
using JuMP, CPLEX
using Graphs
using Dates
using MathOptInterface
const MOI = MathOptInterface

# Defining a constant that controls debug messages
const DEBUG = true # false to deactivate

# The @debug macro for wrapping any code block
macro debug(block)
    return :(if DEBUG
                $(esc(block))
            end)
end


#= Função errada
# Function to read a graph from a file
function read_simple_graph(filepath::String)::SimpleGraph
    """
    Reads an edge list from a file, representing an undirected simple graph.
    Normalizes the vertices to be consecutive 0-based indices
    (internally converted to 1-based for `SimpleGraph`).
    Discards self-loops and multiple edges.
    """
    # read the edges from the file
    edges = open(filepath, "r") do file
        [parse.(Int, split(line)) for line in eachline(file)]
    end

    # Remove loops and multiple edges 
    unique_edges = Set{Tuple{Int,Int}}()
    for (u, v) in edges
        if u != v
            push!(unique_edges, u < v ? (u, v) : (v, u))
        end
    end

    # Create a sorted list of all the present vertices
    all_vertices = sort(collect(unique(vcat(collect.(unique_edges)...))))

    # Creates a normalization map: original -> consecutive index (1-based)
    vertex_map = Dict(v => i for (i, v) in enumerate(all_vertices))

    # Number of normalized vertices
    n = length(all_vertices)

    # Creates a simple graph (1-based)
    g = SimpleGraph(n)

    # Add the normalized edges 
    for (u, v) in unique_edges
        u_norm = vertex_map[u] 
        v_norm = vertex_map[v]
        add_edge!(g, u_norm, v_norm)
    end

    return g
end
=#


# Function to read a graph from a file
function read_simple_graph(filepath::String)::SimpleGraph
    """
    Reads an edge list from a file, representing an undirected simple graph.
    Normalizes the vertices to be consecutive 0-based indices
    (internally converted to 1-based for `SimpleGraph`).
    Discards self-loops and multiple edges.
    """
    # read the edges from the file
    edges = open(filepath, "r") do file
        [parse.(Int, split(line)) for line in eachline(file)]
    end

    # Create a sorted list of all the present vertices (descarta duplicatas)
    all_vertices = sort(collect(unique(vcat(collect.(edges)...))))

    # Creates a normalization map: original -> consecutive index (1-based)
    vertex_map = Dict(v => i for (i, v) in enumerate(all_vertices))

    # Number of normalized vertices
    n = length(all_vertices)

    # Creates a simple empty graph (1-based)
    g = SimpleGraph(n)

    # Remove loops and multiple edges from the edges list 
    unique_edges = Set{Tuple{Int,Int}}()
    for (u, v) in edges
        if u != v
            push!(unique_edges, u < v ? (u, v) : (v, u))
        end
    end

    # Add the normalized edges 
    for (u, v) in unique_edges
        u_norm = vertex_map[u] 
        v_norm = vertex_map[v]
        add_edge!(g, u_norm, v_norm)
    end

    return g
end

# Function that computes a graceful coloring of the input graph
#=
    This function takes as input the simple graph, the maximum time (in minutes)
    allowed for the solver to run on this graph, and a viable solution. If the solver does not 
    find the optimal solution within this time, the best solution found will be returned.
=#
function twoStrongRoman(graph::SimpleGraph, timelimitMinutes::Int)
    # Preparing the optimization model
    model = Model(CPLEX.Optimizer)

    set_optimizer_attribute(model, MOI.Silent(), true) # activate log if false

    # Setting the maximum time (in seconds) for CPLEX running time
    set_optimizer_attribute(model, MOI.TimeLimitSec(), timelimitMinutes*60)

    # ------------------------------------------------------------------------
    # Defining decision variables
    @variable(model, z[v in vertices(graph), k in 0:3], Bin)
    @variable(model, a[u in vertices(graph), v in neighbors(graph, u)], Bin)
    @variable(model, q[u in vertices(graph), k in 0:2], Bin)
    
    # -------------------------------------------------------------------------
    # constraint R1: each vertex receives exactly one label 
    for v in vertices(graph)
        @constraint(model, z[v,0]+z[v,1]+z[v,2]+z[v,3] == 1)
    end

    # constraint R3: each vertex v with f(v) = 2 must have at most one neighbor u 
    # with label 0 such that v is the unique protector of u.
    #for v in vertices(graph)  
    #    if degree(graph, v) > 0
    #        @constraint(model, sum(a[u,v] for u in neighbors(graph, v)) <= z[v,2])
    #    end
    #end

    for u in vertices(graph)
        deg_u = degree(graph, u)
        
        if deg_u == 0
            # Isolated vertex must have label >= 1
            @constraint(model, z[u,0] == 0)
            continue
        end

        # calculates the number of neighbors of u that are protectors
        protector_sum(u) = sum(z[w,2] + z[w,3] for w in neighbors(graph,u))

        # constraint R2: each vertex with label 0 must have at least one protector
        @constraint(model, protector_sum(u) >= z[u,0])

        # constraint R3: each vertex u with f(u) = 2 must have at most one neighbor w 
        # with label 0 such that u is the unique protector of w.
        @constraint(model, sum(a[w,u] for w in neighbors(graph, u)) <= z[u,2])

        # constraints R4.1 to R4.6: guarantees that q[(u,1)] = 1 iff u has exactly one protector
        @constraint(model, q[u,0]+q[u,1]+q[u,2] == 1)                               # R4.1
        @constraint(model, protector_sum(u) <= 0 + deg_u*(1 - q[u,0]))              # R4.2
        @constraint(model, protector_sum(u) <= 1 + deg_u*(1 - q[u,1]))              # R4.3
        @constraint(model, protector_sum(u) >= 1 - deg_u*(1 - q[u,1]))              # R4.4
        @constraint(model, protector_sum(u) >= 2 * q[u,2])                          # R4.5
        @constraint(model, protector_sum(u) <= 1 + (deg_u - 1) * q[u,2])            # R4.6
    end

    # constraints R5 to R8: guarantees that a[u,v] = 1 iff f(u)=0, f(v)=2 and v is the unique protector of u.
    for u in vertices(graph)
        for v in neighbors(graph, u)
            @constraint(model, a[u,v] <= z[u,0])                    # R5
            @constraint(model, a[u,v] <= z[v,2])                    # R6
            @constraint(model, a[u,v] <= q[u,1])                    # R7
            @constraint(model, a[u,v] >= z[u,0]+z[v,2]+q[u,1]-2)    # R8
        end
    end

    # Setting the objective function
    @objective(model, Min, sum(z[v,1]+2*z[v,2]+3*z[v,3] for v in vertices(graph)))

    # Printing the model
    # @debug println(model) 

    @debug begin
        num_vars = JuMP.num_variables(model)
        println("Number of variables of the model: ", num_vars)

        # conta apenas as restrições do modelo e exclui restrições de integralidade e de variáveis binárias
        num_cons = JuMP.num_constraints(model; count_variable_in_set_constraints = false)
        println("Number of restrictions of the model: ", num_cons)
    end

    start_time = now()
    JuMP.optimize!(model)
    end_time = now()

    @debug begin
        println(stderr, "Termination status: ", termination_status(model))
        println(stderr, "Primal status: ", primal_status(model))
        println(stderr, "Dual status: ", dual_status(model))
    end

    status = termination_status(model)

    if has_values(model)
        obj = JuMP.objective_value(model)
        z_star = JuMP.value.(z)
    else
        println(stderr, "Error: solution not found")
        exit(1)
    end

    return z_star, obj, Dates.value(end_time - start_time), status
end

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
function main()
    # Reads the file path from the terminal
    # Checks if the user provided any arguments
    if length(ARGS) < 1
        println(stderr, "Usage: julia roman.jl <arquivo.txt>")
        exit(1)
    end

    # Takes the first argument as the file path
    filepath = ARGS[1]

    # Verify whether the file exists
    if !isfile(filepath)
        println(stderr, "Error: file not found -> ", filepath)
        exit(1)
    end

    # Splits the path and the file extension
    filename, ext = splitext(filepath)

    # Read the graph and runs the solver 
    g = read_simple_graph(filepath)

    #@debug begin
    #    println("graph: ", g)
    #end

    n = nv(g)  
    m = ne(g)  
    density = 2.0*m / (n*(n-1))

    @debug begin
        println("n = $n, m = $m, density = $(density)")
    end

    # quando o grafo é vazio, o valor default será zero
    max_degree = maximum((degree(g, v) for v in vertices(g)); init=0) 
    min_degree = minimum((degree(g, v) for v in vertices(g)); init=0)

    TIME_LIMIT_MINUTES = 5 

    z, opt, elapsed_time, status = twoStrongRoman(g, TIME_LIMIT_MINUTES)

    @debug begin
        println("opt = ", opt)
        for v in vertices(g)
            println("label($v) = $(z[v,1]+2*z[v,2]+3*z[v,3])")
        end
    end

    # Creates the 'output' directory if it does not exist
    output_dir = "output"
    isdir(output_dir) || mkdir(output_dir)

    # Full path of the output CSV file
    result_file_path = joinpath(output_dir, basename(filename) * ".csv")

    # Writing the result in a CSV file
    open(result_file_path, "w") do io
        # Header
        println(io, "G,|V|,|E|,density,maxDegree,minDegree,weight,time(milliseconds),status")
        # Row with results
        println(io, "$(basename(filename)),$(nv(g)),$(ne(g)),$density,$max_degree,$min_degree,$opt,$elapsed_time,$status")
    end

    println(stdout, "Results saved in: $result_file_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
