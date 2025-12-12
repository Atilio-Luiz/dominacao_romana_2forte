# 📘 Graceful Coloring Integer Programming

Integer Programming for Graceful Coloring of Graphs 

---

**Definition:** Let $G = (V(G),E(G))$ be simple graph and let $S = \lbrace 1,2,\ldots,k \rbrace$ be a set of whole numbers called **colors**. A **k-graceful coloring** of $G$ is a function $f \colon V(G) \to \lbrace 1,2,\ldots,k \rbrace$ such that the following two conditions are satisfied:

- every two adjacent vertices $u, v \in V(G)$ are assigned differente colors, that is, $f(u) \neq f(v)$ for every $uv \in E(G)$; and
- when every edge $uv \in E(G)$ is assigned the induced edge label $|f(u)-f(v)|$, we obtain that any two adjacent edges have distinct induced edge labels.

The minimum whole number $k$ for which a graph $G$ admits a $k$-graceful coloring is called the **graceful chromatic number** of $G$ and is denoted by $\chi_g(G)$. The **Graceful Coloring Problem** consists in determining the exact value of the graceful chromatic number of an arbitrary graph. This problem is an NP-hard problem.

In this project, I propose an Integer Programming formulation for the graceful coloring problem and implement it in the framework provided by the Julia language, using CPLEX as the solver.


## ✅ Requirements

Before running the project, make sure you have installed:

- [IBM ILOG CPLEX Optimization Studio](https://www.ibm.com/br-pt/products/ilog-cplex-optimization-studio)
- [Julia](https://julialang.org/) 
- After installing Julia, you must install the following Julia packages:
    - JuMP (optimization modeling)
    - CPLEX (integer programming solver)
    - Graphs.jl (graph manipulation)
    - DelimitedFiles (file reading/writing)
    - Dates (date handling)
    - MathOptInterface (optimization interface)
 

---

## ▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Atilio-Luiz/gracefulColoringIntegerProgramming.git
   cd gracefulColoringIntegerProgramming

2. Make the script *run_all.sh* executable:
    ```bash
    chmod +x run_all.sh

3. Execute the script:
    ```bash
    ./run_all.sh

The script *run_all.sh* is responsible for executing the program  *graceful_coloring_IP_model.jl*   

## Author ✍️
Atilio G. Luiz (gomes.atilio@ufc.br)



## 📝 License

This project is licensed under a license. See the [LICENCE](LICENSE) file for details.