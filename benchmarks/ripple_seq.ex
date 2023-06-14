defmodule BMP do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/bmp_nifs', 0)
  end
  def gen_bmp_nif(_string,_dim,_mat) do
      raise "gen_bmp_nif not implemented"
  end
  def gen_bmp(string,dim,%Matrex{data: matrix} = _a) do
    gen_bmp_nif(string,dim,matrix)
  end
end

defmodule Ripple do

  def gen_pixel(dim,ticks,{y,x}) do
    fx = 0.5 *  x - dim/15;#x - dim/2;
    fy = 0.5 *  y - dim/15;#y - dim/2;
    d  = :math.sqrt( fx * fx + fy * fy );
    grey = floor(128.0 + 127.0 * :math.cos(d/10.0 - ticks/7.0) /(d/10.0 + 1.0));
    [grey,grey,grey,255]

  end
  def ripple_seq(dim, ticks, [{y,x}]) do
    gen_pixel(dim, ticks, {y,x})
  end
  def ripple_seq( dim,ticks ,[{y,x}|tail]) do
    l=ripple_seq(dim,ticks, tail)
    narray = gen_pixel(dim,ticks, {y,x})
    narray ++ l
  end
end

dim = 5000

indices = for i <- Enum.to_list(0..(dim-1)), j<-Enum.to_list(0..(dim-1)), do: {i,j}
#IO.inspect indices
prev = System.monotonic_time()
imagelist= Ripple.ripple_seq(dim,10,indices)
next = System.monotonic_time()
IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"
imageseq=Matrex.new([imagelist])
BMP.gen_bmp('rippleseq.bmp',dim,imageseq)
