#==========================================================================
Implementation of the greedy algorithm proposed in the article: 
Graph protection under multiple simultaneous attacks: A heuristic approach
DOI: https://doi.org/10.1016/j.knosys.2024.112791
===========================================================================#
using Graphs
using GraphPlot, Compose, Cairo

function greedy_solution(graph::SimpleGraph)::Dict{Int,Int}
    weights = Dict{Int,Int}()
    covered = falses(nv(graph))       # Bool array
    neighs  = neighbors               # alias para evitar lookup repetido

    # initialize all weights to 0
    for v in vertices(graph)
        weights[v] = 0
    end

    while count(covered) != nv(graph)
        g = Dict{Int,Int}()

        # compute greedy score g[v]
        for v in vertices(graph)
            if !covered[v]
                counter = 1
                for w in neighs(graph, v)
                    if !covered[w]
                        counter += 1
                    end
                end
                g[v] = counter
            end
        end

        # pick vertex that maximizes g[v]
        v = argmax(g)

        # compute uncov_v without alocar sets
        v_not_counted = covered[v] ? 1 : 0
        weights[v] = min(3, g[v] + v_not_counted)

        # mark neighbors as covered
        covered[v] = true
        for w in neighs(graph, v)
            covered[w] = true
        end
    end

    return weights
end


g = Graphs.SimpleGraphs.petersen_graph()
d = greedy_solution(g)
println(d)