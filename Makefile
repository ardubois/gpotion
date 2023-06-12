all: priv/gpu_nifs.so 

priv/gpu_nifs.so: c_src/gpu_nifs.cu
	nvcc --shared -g --compiler-options '-fPIC' -o priv/gpu_nifs.so c_src/gpu_nifs.cu

bmp: c_src/bmp_nifs.cu 
	nvcc --shared -g --compiler-options '-fPIC' -o priv/bmp_nifs.so c_src/bmp_nifs.cu

clean:
	rm priv/gpu_nifs.so
