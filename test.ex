defmodule MyKernel do
  import GPotion.TypeInference
  import GPotion
kernel2 gpu_nBodies(p,dt,n,softening) do
   i  = blockDim.x * blockIdx.x + threadIdx.x
  if (i < n) do
     fx = 0.0
     fy = 0.0
     fz  = 0.0
    for j in range(0,n) do
       dx = p[6*j] - p[6*i];
       dy = p[6*j+1] - p[6*i+1];
       dz = p[6*j+2] - p[6*i+2];
       distSqr   = dx*dx + dy*dy + dz*dz + softening;
       invDist  = 1.0/sqrt(distSqr);
       invDist3   = invDist * invDist * invDist;

      fx = fx + dx * invDist3;
      fy = fy + dy * invDist3;
      fz = fz + dz * invDist3;
    end
    p[6*i+3] = p[6*i+3]+ dt*fx;
    p[6*i+4] = p[6*i+4]+ dt*fy;
    p[6*i+5] = p[6*i+5]+ dt*fz;
  end

end
end
GPotion.build(&MyKernel.gpu_nBodies/4)
