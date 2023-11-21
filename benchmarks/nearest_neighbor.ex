defmodule DataSet do
  def open_data_set(file) do
    {:ok, contents} = File.read(file)
    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn f ->  load_file(f) end)
    |> Enum.concat()
    |> Enum.concat()
 #   |> Enum.unzip()
  end
  def load_file(file) do
    #IO.puts file
    {:ok, contents} = File.read(file)
    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn line -> words = String.split(line, " ", trim: true)
                           [ elem(Float.parse(Enum.at(words, 6)),0), elem(Float.parse(Enum.at(words,7)), 0) ] end  )
  end
end

defmodule NN do
  import GPotion
  def euclid_seq([],_lat,_lng) do
    []
  end
  def euclid_seq(array,lat,lng) do
     m_lat = Enum.at(array,0)
     m_lng = Enum.at(array,1)
     value = :math.sqrt((lat-m_lat)*(lat-m_lat)+(lng-m_lng)*(lng-m_lng))
     [value|euclid_seq(Enum.drop(array,2),lat,lng)]
  end
  gpotion euclid(d_locations, d_distances, numRecords, lat, lng) do
    globalId = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
    ilat = 2 * globalId
    ilng = (2 * globalId) + 1
    if (globalId < numRecords) do
      d_distances[globalId] = sqrt((lat-d_locations[ilat])*(lat-d_locations[ilat])+(lng-d_locations[ilng])*(lng-d_locations[ilng]))
    end
  end


end
d1 = DataSet.open_data_set("files")
#IO.inspect(d1)

size = div(length(d1),2)

m1 = Matrex.new([d1])

ker=GPotion.load(&NN.euclid/5)

prev = System.monotonic_time()

locations = GPotion.new_gmatrex(m1)
distances = GPotion.new_gmatrex(1,size)

GPotion.spawn(ker,{size,1,1},{1,1,1},[locations,distances,size,0,0])

_dist_result = GPotion.get_gmatrex(distances)

next = System.monotonic_time()
IO.puts "GPotion\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

IO.puts "start Sequential!"
prev = System.monotonic_time()
_r_sequential = NN.euclid_seq(d1,0,0)
next = System.monotonic_time()
IO.puts "Elixir\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"


#IO.inspect(m1)

