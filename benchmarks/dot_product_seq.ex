
[arg] = System.argv()
n = String.to_integer(arg)

a = Matrex.new([Enum.to_list(1..n)])
b = Matrex.new([Enum.to_list(1..n)])

prev = System.monotonic_time()
_c = Matrex.dot_nt(a, b)
next = System.monotonic_time()

IO.puts "Elixir\t#{n}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
