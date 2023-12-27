for i in $(seq 1 30)
do
        mix run benchmarks/raytracer.ex 5120
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer.ex 7168
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer.ex 9216
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer.ex 11264
done
for i in $(seq 1 30)
do
        benchmarks/cuda/raytracer 5120
done
for i in $(seq 1 30)
do
        benchmarks/cuda/raytracer 7168
done
for i in $(seq 1 30)
do
        benchmarks/cuda/raytracer 9216
done
for i in $(seq 1 30)
do
        benchmarks/cuda/raytracer 11264
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer_seq.ex 5120
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer_seq.ex 7168
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer_seq.ex 9216
done
for i in $(seq 1 30)
do
        mix run benchmarks/raytracer_seq.ex 11264
done