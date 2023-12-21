
[arg] = System.argv()
n = String.to_integer(arg)

prev = System.monotonic_time()
_c = Enum.reduce(Enum.to_list(0..(n-1)),0, fn (a , b) -> a + b end)
next = System.monotonic_time()
IO.puts "Elixir\t#{n}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
