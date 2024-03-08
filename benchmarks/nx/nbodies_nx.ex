#Mix.install([{:exla, "~> 0.7.1"}])

Nx.global_default_backend(EXLA.Backend)

defmodule NBodies do
  import Nx.Defn
  defn integrate(p,dt) do
    case Nx.shape(p) do
      {6} -> :ok
      _ -> raise "invalid shape"
    end
    x = p[0]
    y = p[1]
    z = p[2]
    vx = p[3]
    vy = p[4]
    vz = p[5]

  x = x + vx * dt
  y = y + vy * dt
  z = z + vz * dt
  Nx.stack([x,y,z,vx,vy,vz])
  end
  defn nbodies(p,p_copy,dt,softening) do

      case Nx.shape(p) do
        {6} -> :ok
        _ -> raise "invalid shape"
      end
      x = p[0]
      y = p[1]
      z = p[2]
      vx = p[3]
      vy = p[4]
      vz = p[5]

      z1 = vz - vz

      {fx,fy,fz} = calc_nbodies(p_copy,softening,x,y,z,z1,z1,z1)

      print_value fx

      vx = vx + dt*fx
      vy = vy  + dt*fy
      vz = vz + dt*fz

      Nx.stack([x,y,z,vx,vy,vz])

  end
  defn calc_nbodies(p,softening,x,y,z,fx,fy,fz) do


    {_softening,_x,_y,_z,fx,fy,fz}=while {softening,x,y,z,fx,fy,fz}, i <- p do
      case Nx.shape(i) do
        {6} -> :ok
        _ -> raise "invalid shape"
      end
      jx = i[0]
      jy = i[1]
      jz = i[2]
      dx = jx - x
      dy = jy - y
      dz = jz - z

      distSqr = dx*dx + dy*dy + dz*dz + softening;
      #raise "#{inspect_value(distSqr)}"
      invDist = 1/Nx.sqrt(distSqr);
      invDist3 = invDist * invDist * invDist;

       fx = fx + dx * invDist3;
    fy = fy + dy * invDist3;
    fz = fz + dz * invDist3;
    {softening,x,y,z,fx,fy,fz}
    end
    {fx,fy,fz}
  end
 # defn teste1(p) do
 #   case Nx.shape(p) do
 #     {6} -> :ok
 #     _ -> raise "invalid shape"
 #   end
 #   x = p[0]
 #   y = p[1]
 #   z = p[2]
 #   vx = p[3]
 #   vy = p[4]
 #   vz = p[5]
 #   Nx.stack([x,y])
 # end
 # defn teste2(p,x,y,z,fx,fy,fz) do
 #   while {x,y,z,fx,fy,fz}, i <- p do
 #     fx1 = i[0]
 #     fy1 = i[1]
 #     fz1 = i[2]

 #   {x,y,z,fx1+fx,fy1+fy,fz1+fz}
  #  end
  #end
end

[arg] = System.argv()

size_matrex = String.to_integer(arg)

softening = 0.000000001;
dt = 0.01; # time step

#size_matrex = 2000

h_buf = Matrex.random(size_matrex,6)

#IO.inspect h_buf
h_buf =Matrex.to_list_of_lists(h_buf)

#IO.inspect h_buf

tensor = Nx.tensor(h_buf, type: :f32)

#t = Nx.tensor([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]], type: :f32)

#t2 = Nx.vectorize(t, :coords)

#IO.inspect tensor

t2 =  Nx.vectorize(tensor, :coords)
prev = System.monotonic_time()
r = NBodies.nbodies(t2,tensor, dt, softening)
_r=NBodies.integrate(r,dt)
next = System.monotonic_time()
IO.puts "Nx\t#{size_matrex}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

#IO.inspect r

#r = NBodies.teste1(t2)

#r = NBodies.teste2(t,Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32))

#r = NBodies.calc_nbodies(t,softening,Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32),Nx.tensor(0,type: :f32))


#IO.inspect r

#NBodies.nbodies(t2,t,dt,softening)
