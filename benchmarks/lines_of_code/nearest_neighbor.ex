require Integer
defmodule DataSet do
  def open_data_set(file) do
    {:ok, contents} = File.read(file)
    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn f ->  load_file(f) end)
    |> Enum.concat()
    |> Enum.concat()
  end
  def load_file(file) do
    #IO.puts file
    {:ok, contents} = File.read(file)
    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn line -> words = String.split(line, " ", trim: true)
                           [ elem(Float.parse(Enum.at(words, 6)),0), elem(Float.parse(Enum.at(words,7)), 0) ] end  )
  end
  def gen_lat_long(_l,c) do
    if(Integer.is_even(c)) do
      (Enum.random(0..358)) + :rand.uniform()
    else
      (7 + Enum.random(0..63)) + :rand.uniform()
    end
  end
end
defmodule NN do
  import GPotion
  gpotion euclid(d_locations, d_distances, numRecords, lat, lng) do
    globalId = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
    ilat = 2 * globalId
    ilng = (2 * globalId) + 1
    if (globalId < numRecords) do
      d_distances[globalId] = sqrt((lat-d_locations[ilat])*(lat-d_locations[ilat])+(lng-d_locations[ilng])*(lng-d_locations[ilng]))
    end
  end
end
[arg] = System.argv()
usr_size = String.to_integer(arg)
size = usr_size
m1 = Matrex.new(1,2*usr_size,&DataSet.gen_lat_long/2)
ker=GPotion.load(&NN.euclid/5)
prev = System.monotonic_time()
locations = GPotion.new_gmatrex(m1)
distances = GPotion.new_gmatrex(1,size)
GPotion.spawn(ker,{size,1,1},{1,1,1},[locations,distances,size,0,0])
_dist_result = GPotion.get_gmatrex(distances)
next = System.monotonic_time()
IO.puts "GPotion\t#{usr_size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
