defmodule GPotion do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/gpu_nifs', 0)
      #IO.puts("ok")
  end
  defp gen_para(p,:matrex) do
    "float *#{p}"
  end
  defp gen_para(p,:float) do
    "float #{p}"
  end
  defp gen_para(p,:int) do
    "int #{p}"
  end
  defmacro kernel(header, do: body) do
      {fname, _, para} = header
     {param_list,types} = if is_list(List.last(para)) do
        types = List.last(para)
        param_list = para
          |> List.delete_at(length(para)-1)
          |> Enum.map(fn({p, _, _}) -> p end)
          |> Enum.zip(types)
          |> Enum.map(fn({p,t}) -> gen_para(p,t) end)
          |> Enum.join(", ")
        {param_list,types}
      else
      types = List.duplicate(:matrex,length(para))
      param_list = para
      |> Enum.map(fn({p, _, _}) -> p end)
      |> Enum.zip(types)
      |> Enum.map(fn({p,t}) -> gen_para(p,t) end)
      |> Enum.join(", ")
      {param_list,types}
     end
     cuda_body = GPotion.CudaBackend.gen_body(body)
     k = GPotion.CudaBackend.gen_kernel(fname,param_list,cuda_body)
     accessfunc = GPotion.CudaBackend.gen_kernel_call(fname,length(types),Enum.reverse(types))
     file = File.open!("c_src/#{fname}.cu", [:write])
     IO.write(file, "#include \"erl_nif.h\"\n\n" <> k <> "\n\n" <> accessfunc)
     File.close(file)
     #IO.puts k
     #IO.puts accessfunc
     quote do
        def unquote(header)do
          raise "A kernel can only be executed with GPotion.spawn"
        end
      end
  end
  defmacro gpmodule(header,do: body) do
    IO.inspect header
    IO.inspect body
  end
  defmacro gpotion(header, do: body) do
    {fname, comp_info, para} = header

    caller_st = __CALLER__
    module_name = to_string caller_st.module
    IO.puts module_name

   {param_list,types_para,is_typed,inf_types} = if is_list(List.last(para)) do
      types_para = List.last(para)
      param_list = para
        |> List.delete_at(length(para)-1)
        |> Enum.map(fn({p, _, _}) -> p end)
        |> Enum.zip(types_para)
        |> Enum.map(fn({p,t}) -> gen_para(p,t) end)
        |> Enum.join(", ")
      {param_list,types_para,true,%{}}
    else
      types = para
      |> Enum.map(fn({p, _, _}) -> p end)
      |> Map.new(fn x -> {x,:none} end)
      |> GPotion.TypeInference.infer_types(body)

      param_list = para
      |> Enum.map(fn {p, _, _}-> gen_para(p,Map.get(types,p)) end)
      |> Enum.join(", ")

      types_para = para
      |>  Enum.map(fn {p, _, _}-> Map.get(types,p) end)
     {param_list,types_para,false,types}

   end
   cuda_body = GPotion.CudaBackend.gen_cuda(body,inf_types,is_typed)
   k = GPotion.CudaBackend.gen_kernel(fname,param_list,cuda_body)
   accessfunc = GPotion.CudaBackend.gen_kernel_call(fname,length(types_para),Enum.reverse(types_para))
   if(File.exists?("c_src/#{module_name}.cu")) do
    file = File.open!("c_src/#{module_name}.cu", [:append])
    IO.write(file, "\n" <> k <> "\n\n" <> accessfunc)
  else
    file = File.open!("c_src/#{module_name}.cu", [:write])
    IO.write(file, "#include \"erl_nif.h\"\n\n" <> k <> "\n\n" <> accessfunc)
    File.close(file)
  end
   #IO.puts k
   #IO.puts accessfunc
   para = if is_list(List.last(para)) do List.delete_at(para,length(para)-1) else para end
   para = para
    |> Enum.map(fn {p, b, c}-> {String.to_atom("_" <> to_string(p)),b,c} end)

  {_result, _errcode} = System.cmd("nvcc",
  ["--shared",
  "--compiler-options",
  "'-fPIC'",
  "-o",
  "priv/#{module_name}.so",
  "c_src/#{module_name}.cu"
  ], stderr_to_stdout: true)
  File.rename("c_src/#{module_name}.cu","c_src/#{module_name}_gp.cu")

   quote do
      def unquote({fname,comp_info, para})do
        raise "A kernel can only be executed with GPotion.spawn"
      end
    end
end

  def create_ref_nif(_matrex) do
    raise "NIF create_ref_nif/1 not implemented"
end
def new_gmatrex(%Matrex{data: matrix} = a) do
  ref=create_ref_nif(matrix)
  {ref, Matrex.size(a)}
end
def new_gmatrex(r,c) do
  ref=new_ref_nif(c)
  {ref, {r,c}}
  end
def new_ref_nif(_matrex) do
  raise "NIF new_ref_nif/1 not implemented"
end
def synchronize_nif() do
  raise "NIF new_ref_nif/1 not implemented"
end
def synchronize() do
  synchronize_nif()
end
def new_ref(size) do
ref=new_ref_nif(size)
{ref, {1,size}}
end
def get_matrex_nif(_ref,_rows,_cols) do
raise "NIF get_matrex_nif/1 not implemented"
end
def get_gmatrex({ref,{rows,cols}}) do
%Matrex{data: get_matrex_nif(ref,rows,cols)}
end

def load_kernel_nif(_module,_fun) do
  raise "NIF new_ref_nif/2 not implemented"
end
def load(kernel) do
  case Macro.escape(kernel) do
    {:&, [],[{:/, [], [{{:., [], [module, kernelname]}, [no_parens: true], []}, _nargs]}]} ->


              #IO.puts module
              #raise "hell"
              GPotion.load_kernel_nif(to_charlist(module),to_charlist(kernelname))

    _ -> raise "GPotion.build: invalid kernel"
  end
end
def spawn_nif(_k,_t,_b,_l) do
  raise "NIF spawn_nif/1 not implemented"
end
def spawn(k,t,b,l) when is_function(k) do
  load(k)
  spawn_nif(k,t,b,Enum.map(l,&get_ref/1))
end
def spawn(k,t,b,l) do
  spawn_nif(k,t,b,Enum.map(l,&get_ref/1))
end
def get_ref({ref,{_rows,_cols}}) do
  ref
end
def get_ref(e) do
  e
end
end
