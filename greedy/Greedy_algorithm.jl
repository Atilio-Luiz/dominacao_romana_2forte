#==========================================================================
Implementation of the greedy algorithm proposed in the article: 
Graph protection under multiple simultaneous attacks: A heuristic approach
DOI: https://doi.org/10.1016/j.knosys.2024.112791
===========================================================================#
using Graphs
using Random

function greedy_solution(graph::SimpleGraph)::Dict{Int,Int}
    weights = Dict{Int,Int}()
    covered = falses(nv(graph))       # boolean vector to mark the covered vertices
    neighs  = neighbors               # alias

    # initialize all weights to 0
    for v in vertices(graph)
        weights[v] = 0
    end

    while count(covered) != nv(graph)
        g = Dict{Int,Int}()

        # compute greedy score g[v] for all the vertices of the graph
        for v in vertices(graph)
            counter = 0
            if !covered[v]
                counter = 1
            end
            for w in neighs(graph, v)
                if !covered[w]
                    counter += 1
                end
            end
            g[v] = counter
        end

        # pick vertex that maximizes g[v]
        max_g = maximum(values(g))

        # candidatos com g[v] máximo
        candidates = [v for (v,gv) in g if gv == max_g]

        # tenta desempatar preferindo vértices não cobertos
        uncovered_candidates = [v for v in candidates if !covered[v]]

        shuffle!(uncovered_candidates)

        v = isempty(uncovered_candidates) ? first(candidates) : first(uncovered_candidates)

        # compute uncov_v without alocate sets
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


#g = Graphs.SimpleGraphs.petersen_graph()
#d = greedy_solution(g)
#println(d)