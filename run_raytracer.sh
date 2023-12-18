for i in $(seq 1 30)
do
        mix run benchmarks/raytracer.ex 1024
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer.ex 2048
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer.ex 3072
done
for i in $(seq 1 30)
do
        mix run benchmarks/cuda/raytracer 1024
done
for i in $(seq 1 30)
do
        mix run benchmarks/cuda/raytracer 2048
done
for i in $(seq 1 30)
do
        mix run benchmarks/cuda/raytracer 3072
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer_seq.ex 1024
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer_seq.ex 2048
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer_seq.ex 3072
done