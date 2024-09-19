import Bitwise
defmodule RayTracer do
import GPotion
gpotion raytracing(width, height, spheres, image) do
  x = threadIdx.x + blockIdx.x * blockDim.x
  y = threadIdx.y + blockIdx.y * blockDim.y
  offset = x + y * blockDim.x * gridDim.x
  ox = (x - width/2)
  oy = (y - height/2)
   maxz = -99999.0
  for i in range(0, 20) do
    sphereRadius = spheres[i * 7 + 3]
    dx = ox - spheres[i * 7 + 4]
    dy = oy - spheres[i * 7 + 5]
    n = 0.0
    t = -99999.0
    dz = 0.0
    if (dx * dx + dy * dy) <  (sphereRadius * sphereRadius) do
      dz = sqrtf(sphereRadius * sphereRadius - (dx * dx) - (dy * dy))
      n = dz / sqrtf(sphereRadius * sphereRadius)
      t = dz + spheres[i * 7 + 6]
    else
      t = -99999.0
      n = 0.0
    end
    if t > maxz do
      fscale = n
      r = spheres[i * 7 + 0] * fscale
      g = spheres[i * 7 + 1] * fscale
      b = spheres[i * 7 + 2] * fscale
      maxz = t
    end
  end
  image[offset * 4 + 0] = r * 255
  image[offset * 4 + 1] = g * 255
  image[offset * 4 + 2] = b * 255
  image[offset * 4 + 3] = 255
end
end
defmodule Main do
    def rnd(x),  do: x * Random.randint(1, 32767) / 32767
    def sphereMaker2(0, _radius, _sum), do: []
    def sphereMaker2(n, radius, sum) do
      [ Main.rnd(1),
        Main.rnd(1),
        Main.rnd(1),
        Main.rnd(radius) + sum,
        Main.rnd(Main.dim) - trunc(Main.dim / 2),
        Main.rnd(Main.dim) - trunc(Main.dim / 2),
        Main.rnd(256) - 128
      | sphereMaker2(n - 1, radius, sum)]
    end
    def dim, do: String.to_integer(hd(System.argv()))
    def main do
        spheres = 20
        {radius, sum} = {160,20}
        sphereList = Matrex.new([sphereMaker2(spheres, radius, sum)])
        width = Main.dim
        height = width
        prev = System.monotonic_time()
        refSphere = GPotion.new_gmatrex(sphereList)
        refImag = GPotion.new_gmatrex(1, (width)  * (height)  * 4)
        GPotion.spawn(&RayTracer.raytracing/4,{trunc(width/16),trunc(height/16),1},{16,16,1},[width, height, refSphere, refImag])
        GPotion.synchronize()
        _image = GPotion.get_gmatrex(refImag)
        next = System.monotonic_time()
        IO.puts "GPotion\t#{width}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "
    end
end
Main.main
