defmodule NBodies do
  def enbodies(-1,p,_dt,_softening,_n) do
    p
  end
  def enbodies(i,p,dt,softening,n) do
    #p=nbodies(i-1,p,dt,softening,n)
    {fx,fy,fz} = ecalc_nbodies(n,i,p,softening,0.0,0.0,0.0)

    p=Matrex.set(p,1,6*i+4,Matrex.at(p,1,6*i+4)+ dt*fx);
    p=Matrex.set(p,1,6*i+5,Matrex.at(p,1,6*i+5) + dt*fy);
    p=Matrex.set(p,1,6*i+6,Matrex.at(p,1,6*i+6) + dt*fz);
    enbodies(i-1,p,dt,softening,n)
  end

def ecalc_nbodies(-1,_i,_p,_softening,fx,fy,fz) do
  {fx,fy,fz}
end
def ecalc_nbodies(j,i,p,softening,fx,fy,fz) do
    dx = Matrex.at(p,1,(6*j)+1) - Matrex.at(p,1,(6*i)+1);
    dy = Matrex.at(p,1,(6*j)+2) - Matrex.at(p,1,(6*i)+2);
    dz = Matrex.at(p,1,(6*j)+3) - Matrex.at(p,1,(6*i)+3);
    distSqr = dx*dx + dy*dy + dz*dz + softening;
    invDist = 1/:math.sqrt(distSqr);
    invDist3 = invDist * invDist * invDist;

    fx = fx + dx * invDist3;
    fy = fy + dy * invDist3;
    fz = fz + dz * invDist3;
    ecalc_nbodies(j-1,i,p,softening,fx,fy,fz)
end

def nbodies([],_p_copy,_dt,_softening,_n, resp) do
  Enum.reverse(resp)
end
def nbodies(p,p_copy,dt,softening,n,resp) do
  #p=nbodies(i-1,p,dt,softening,n)
  [x,y,z,vx,vy,vz] = Enum.take(p,6)
  rest_p = Enum.drop(p,6)
  {fx,fy,fz} = calc_nbodies(p_copy,softening,x,y,z,0.0,0.0,0.0)

  vx = vx + dt*fx
  vy = vy  + dt*fy
  vz = vz + dt*fz

  nbodies(rest_p,p_copy,dt,softening,n, [x,y,z,vx,vy,vz] ++ resp)
end

def calc_nbodies([],_softening,_x,_y,_z,fx,fy,fz) do
  {fx,fy,fz}
end
def calc_nbodies(p,softening,x,y,z,fx,fy,fz) do
    [jx,jy,jz,_jvx,_jvy,_jvz] = Enum.take(p,6)
    rest_p = Enum.drop(p,6)
    dx = jx - x
    dy = jy - y
    dz = jz - z
    distSqr = dx*dx + dy*dy + dz*dz + softening;
    invDist = 1/:math.sqrt(distSqr);
    invDist3 = invDist * invDist * invDist;

    fx = fx + dx * invDist3;
    fy = fy + dy * invDist3;
    fz = fz + dz * invDist3;
    calc_nbodies(rest_p,softening,x,y,z,fx,fy,fz)
end

def ecpu_integrate(-1,p,_dt) do
  p
end
def ecpu_integrate(i,p, dt) do
      p=Matrex.set(p,1,6*i+1,Matrex.at(p,1,6*i+1) + Matrex.at(p,1,6*i+4)*dt)
      p=Matrex.set(p,1,6*i+2,Matrex.at(p,1,6*i+2) + Matrex.at(p,1,6*i+5)*dt)
      p=Matrex.set(p,1,6*i+3,Matrex.at(p,1,6*i+3) + Matrex.at(p,1,6*i+6)*dt)
      ecpu_integrate(i-1,p,dt)
end
def cpu_integrate([],_dt,resp) do
  Enum.reverse(resp)
end
def cpu_integrate(p, dt,result) do
  [x,y,z,vx,vy,vz]= Enum.take(p,6)
  rest_p = Enum.drop(p,6)
  x = x + vx * dt
  y = y + vy * dt
  z = z + vz * dt
  cpu_integrate(rest_p,dt, [x,y,z,vx,vy,vz]++result)
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


nBodies = user_value

softening = 0.000000001;
dt = 0.01; # time step

size_matrex = 6 * nBodies


h_buf = Matrex.random(1,size_matrex)
ebuf =Matrex.to_list h_buf

prev = System.monotonic_time()
lresp=NBodies.nbodies(ebuf,ebuf,dt,softening,nBodies,[])
_lresp = NBodies.cpu_integrate(lresp,dt,[])
next = System.monotonic_time()
IO.puts "Elixir\t#{nBodies}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

##mresp = Matrex.new([lresp])

##prev = System.monotonic_time()
##cpu_resp = NBodies.enbodies(nBodies-1,h_buf,dt,softening,nBodies-1)
##cpu_resp = NBodies.ecpu_integrate(nBodies-1,cpu_resp,dt)
##next = System.monotonic_time()
##IO.puts "Elixir\t#{nBodies}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

##NBodies.check_equality(nBodies,cpu_resp,mresp)
