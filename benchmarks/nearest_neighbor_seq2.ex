require Integer
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
  def gen_data_set(n), do: gen_data_set_(n,[])
  def gen_data_set_(0,data), do: data
  def gen_data_set_(n,data) do
    lat = (7 + Enum.random(0..63)) + :rand.uniform();
      lon = (Enum.random(0..358)) + :rand.uniform();
      gen_data_set_(n-1, [lat,lon|data])

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
  def euclid_seq(a,lat,lng) do
    {_l,col} = Matrex.size(a)
    c = div(col,2)
    euclid_seq_(c-1, a,lat,lng,[])
  end
  def euclid_seq_(0,array,lat,lng, data) do
     m_lat = Matrex.at(array,1,2*0+1)
     m_lng = Matrex.at(array,1,2*0+2)
          #m_lng = Enum.at(array,1)
     value = :math.sqrt((lat-m_lat)*(lat-m_lat)+(lng-m_lng)*(lng-m_lng))
     [value|data]
  end
  def euclid_seq_(n,array,lat,lng,data) do
   # IO.puts n
     m_lat = Matrex.at(array,1,2*n+1)
     m_lng = Matrex.at(array,1,2*n+2)
          #m_lng = Enum.at(array,1)
     value = :math.sqrt((lat-m_lat)*(lat-m_lat)+(lng-m_lng)*(lng-m_lng))
     euclid_seq_(n-1,array,lat,lng,[value|data])
  end




end
#d1 = DataSet.open_data_set("files")
#IO.inspect(d1)

#d1 = DataSet.gen_data_set(100000000)

[arg] = System.argv()

usr_size = String.to_integer(arg)
#d1 = DataSet.gen_data_set(usr_size)


#size = usr_size

m1 = Matrex.new(1,2*usr_size,&DataSet.gen_lat_long/2)


prev = System.monotonic_time()
_r_sequential = NN.euclid_seq(m1,0,0)
next = System.monotonic_time()
IO.puts "Elixir\t#{usr_size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"


#IO.inspect(m1)
