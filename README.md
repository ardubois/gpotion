# GPotion

**A DSL for GPU programming in Elixir**

Paper: [GPotion: An embedded DSL for GPU programming in Elixir](https://sol.sbc.org.br/index.php/sblp/article/view/28326) DU BOIS, Andre Rauber; CAVALHEIRO, Gerson. GPotion: An embedded DSL for GPU programming in Elixir. In: SIMPÓSIO BRASILEIRO DE LINGUAGENS DE PROGRAMAÇÃO (SBLP), 27. , 2023, Campo Grande/MS. Anais [...]. Porto Alegre: Sociedade Brasileira de Computação, 2023 . p. 1–8.


## Installation

* Install [Erlang](https://www.erlang.org/), [Elixir](https://elixir-lang.org/install.html) and [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
* Install erlang-dev
* Clone this repository
* cd gpotion
* type: **make clean**
* type: **make**
* Some benchmarks also need: **make bmp**
* type: **mix deps.get**
* type: **mix compile**
* benchmarks are in the ***benchmarks*** folder
* type **mix run benchmakrs/mm.ex 1024**



