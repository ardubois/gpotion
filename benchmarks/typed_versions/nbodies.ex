defmodule NBodies do
  import GPotion
  gpotion gpu_nBodies(p,dt,n,softening,[:matrex,:float,:int,:float]) do
    var i int = blockDim.x * blockIdx.x + threadIdx.x
    if (i < n) do
      var fx float= 0.0
      var fy float= 0.0
      var fz float = 0.0
      for j in range(0,n) do
        var dx float= p[6*j] - p[6*i];
        var dy float= p[6*j+1] - p[6*i+1];
        var dz float= p[6*j+2] - p[6*i+2];
        var distSqr  float = dx*dx + dy*dy + dz*dz + softening;
        var invDist float = 1/sqrt(distSqr);
        var invDist3  float = invDist * invDist * invDist;

        fx = fx + dx * invDist3;
        fy = fy + dy * invDist3;
        fz = fz + dz * invDist3;
      end
      p[6*i+3] = p[6*i+3]+ dt*fx;
      p[6*i+4] = p[6*i+4]+ dt*fy;
      p[6*i+5] = p[6*i+5]+ dt*fz;
    end

  end

  def nbodies(-1,p,_dt,_softening,_n) do
    p
  end
  def nbodies(i,p,dt,softening,n) do
    p=nbodies(i-1,p,dt,softening,n)
    {fx,fy,fz} = calc_nbodies(n,i,p,softening,0.0,0.0,0.0)

    p=Matrex.set(p,1,6*i+4,Matrex.at(p,1,6*i+4)+ dt*fx);
    p=Matrex.set(p,1,6*i+5,Matrex.at(p,1,6*i+5) + dt*fy);
    Matrex.set(p,1,6*i+6,Matrex.at(p,1,6*i+6) + dt*fz);

  end

def calc_nbodies(-1,_i,_p,_softening,fx,fy,fz) do
  {fx,fy,fz}
end
def calc_nbodies(j,i,p,softening,fx,fy,fz) do
    dx = Matrex.at(p,1,(6*j)+1) - Matrex.at(p,1,(6*i)+1);
    dy = Matrex.at(p,1,(6*j)+2) - Matrex.at(p,1,(6*i)+2);
    dz = Matrex.at(p,1,(6*j)+3) - Matrex.at(p,1,(6*i)+3);
    distSqr = dx*dx + dy*dy + dz*dz + softening;
    invDist = 1/:math.sqrt(distSqr);
    invDist3 = invDist * invDist * invDist;

    fx = fx + dx * invDist3;
    fy = fy + dy * invDist3;
    fz = fz + dz * invDist3;
    calc_nbodies(j-1,i,p,softening,fx,fy,fz)
end
gpotion gpu_integrate(p, dt, n,[:matrex,:float,:int]) do
  var i int  = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) do
    p[6*i] = p[6*i] + p[6*i+3]*dt;
    p[6*i+1] = p[6*i+1] + p[6*i+4]*dt;
    p[6*i+2] = p[6*i+2] + p[6*i+5]*dt;
  end
end
def cpu_integrate(-1,p,_dt) do
  p
end
def cpu_integrate(i,p, dt) do
      p=Matrex.set(p,1,6*i+1,Matrex.at(p,1,6*i+1) + Matrex.at(p,1,6*i+4)*dt)
      p=Matrex.set(p,1,6*i+2,Matrex.at(p,1,6*i+2) + Matrex.at(p,1,6*i+5)*dt)
      p=Matrex.set(p,1,6*i+3,Matrex.at(p,1,6*i+3) + Matrex.at(p,1,6*i+6)*dt)
      cpu_integrate(i-1,p,dt)
end
def equality(a, b) do
  if(abs(a-b) < 0.01) do
    true
  else
    false
  end
end
def check_equality(0,_cpu,_gpu) do
  :ok
end
def check_equality(n,cpu,gpu) do
  gpu1 =Matrex.at(gpu,1,n)
  cpu1 = Matrex.at(cpu,1,n)
  if(equality(gpu1,cpu1)) do
    check_equality(n-1,cpu,gpu)
  else
    IO.puts "#{n}: cpu = #{cpu1}, gpu = #{gpu1}"
    check_equality(n-1,cpu,gpu)
  end
end
end

[arg] = System.argv()

user_value = String.to_integer(arg)

nBodies = user_value #3000;
block_size =  128;
nBlocks = floor ((nBodies + block_size - 1) / block_size)
softening = 0.000000001;
dt = 0.01; # time step
size_body = 6

size_matrex = size_body * nBodies


h_buf = Matrex.random(1,size_matrex)




ker1=GPotion.load(&NBodies.gpu_nBodies/4)
ker2=GPotion.load(&NBodies.gpu_integrate/3)

prev = System.monotonic_time()

d_buf =GPotion.new_gmatrex(h_buf)


GPotion.spawn(ker1,{nBlocks,1,1},{block_size,1,1},[d_buf,dt,nBodies,softening])
GPotion.synchronize()
GPotion.spawn(ker2,{nBlocks,1,1},{block_size,1,1},[d_buf,dt,nBodies])
gpu_resp = GPotion.get_gmatrex(d_buf)
next = System.monotonic_time()
IO.puts "gpotion\t#{user_value}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

#IO.inspect gpu_resp

prev = System.monotonic_time()
cpu_resp = NBodies.nbodies(nBodies-1,h_buf,dt,softening,nBodies-1)
cpu_resp = NBodies.cpu_integrate(nBodies-1,cpu_resp,dt)
next = System.monotonic_time()
IO.puts "Elixir\t#{user_value}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

IO.inspect cpu_resp

NBodies.check_equality(nBodies,cpu_resp,gpu_resp)
