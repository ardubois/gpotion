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
  defmacro gptype({func,_,[type]}) do
    if (nil == Process.whereis(:gptype_server)) do
      pid = spawn_link(fn -> gptype_server() end)
      Process.register(pid, :gptype_server)
    end
    send(:gptype_server,{:add_type, func,type_to_list(type)})
    #IO.inspect(type_to_list(type))
    quote do
    end
  end
  def gptype_server(), do: gptype_server_(Map.new())
  defp gptype_server_(map) do
    receive do
      {:add_type, fun, types}  -> map=Map.put(map,fun, types)
                              gptype_server_(map)
      {:get_type, pid,fun} -> type=Map.get(map,fun)
                              send(pid,{:type,fun,type})
                              gptype_server_(map)
      {:kill}               -> :dead
    end
  end
  defp type_to_list({:integer,_,_}), do: [:int]
  defp type_to_list({:unit,_,_}), do: [:unit]
  defp type_to_list({:float,_,_}), do: [:float]
  defp type_to_list({:gmatrex,_,_}), do: [:matrex]
  defp type_to_list({:~>,_, [a1,a2]}), do: type_to_list(a1) ++ type_to_list(a2)
  defp type_to_list({x,_,_}), do: raise "Unknown type constructor #{x}"
  def is_typed?() do
    nil != Process.whereis(:gptype_server)
  end
  def get_type_kernel(fun_name) do
    send(:gptype_server,{:get_type, self(),fun_name})
    receive do
      {:type,fun,type} -> if fun == fun_name do
                                send(:gptype_server,{:kill})
                                type
                          else
                                raise "Asked for #{fun_name} got #{fun}"
                          end
      end

    end
    def get_type_fun(fun_name) do
      send(:gptype_server,{:get_type, self(),fun_name})
      receive do
        {:type,fun,type} -> if fun == fun_name do
                                  type
                            else
                                  raise "Asked for #{fun_name} got #{fun}"
                            end
        end

      end

  defmacro gpotion(header, do: body) do
    {fname, comp_info, para} = header

    caller_st = __CALLER__
    module_name = to_string caller_st.module

    {delta,is_typed}  = if(is_typed?()) do
              types = get_type_kernel(fname)
              delta= para
                |> Enum.map(fn({p, _, _}) -> p end)
                |> Enum.zip(types)
                |> Map.new()
              {delta,true}
            else
              delta=para
                |> Enum.map(fn({p, _, _}) -> p end)
                |> Map.new(fn x -> {x,:none} end)
              {delta,false}
            end




   #inf_types = GPotion.TypeInference.infer_types(delta,body)

   inf_types = GPotion.TypeInference.type_check(delta,body)

   param_list = para
      |> Enum.map(fn {p, _, _}-> gen_para(p,Map.get(inf_types,p)) end)
      |> Enum.join(", ")

   types_para = para
      |>  Enum.map(fn {p, _, _}-> Map.get(inf_types,p) end)

   #{param_list,types_para,is_typed,inf_types} = if is_list(List.last(para)) do
   #   types_para = List.last(para)
   #   param_list = para
   #     |> List.delete_at(length(para)-1)
   #     |> Enum.map(fn({p, _, _}) -> p end)
   #     |> Enum.zip(types_para)
   #     |> Enum.map(fn({p,t}) -> gen_para(p,t) end)
   #     |> Enum.join(", ")
   #   {param_list,types_para,true,%{}}
   # else
   #   types = para
   #   |> Enum.map(fn({p, _, _}) -> p end)
   #   |> Map.new(fn x -> {x,:none} end)
   #   |> GPotion.TypeInference.infer_types(body)
   #
   #   param_list = para
   #   |> Enum.map(fn {p, _, _}-> gen_para(p,Map.get(types,p)) end)
   #   |> Enum.join(", ")
#
 #     types_para = para
  #    |>  Enum.map(fn {p, _, _}-> Map.get(types,p) end)
   #  {param_list,types_para,false,types}
#   end

   inf_types = if is_typed do %{} else inf_types end
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

  {result, errcode} = System.cmd("nvcc",
  ["--shared",
  "--compiler-options",
  "'-fPIC'",
  "-o",
  "priv/#{module_name}.so",
  "c_src/#{module_name}.cu"
  ], stderr_to_stdout: true)

  if errcode == 1 do raise "Error when compiling .cu file generated by GPotion:\n#{result}" end

  File.rename("c_src/#{module_name}.cu","c_src/#{module_name}_gp.cu")

   quote do
      def unquote({fname,comp_info, para})do
        raise "A kernel can only be executed with GPotion.spawn"
      end
    end
end

defmacro defgp(header, do: body) do
  {fname, comp_info, para} = header

  caller_st = __CALLER__
  module_name = to_string caller_st.module
  #IO.puts module_name

  {delta,is_typed,fun_type}  = if(is_typed?()) do
    #IO.puts "asdf"
    types = get_type_fun(fname)
    [fun_type|_] = Enum.reverse(types)
    delta= para
      |> Enum.map(fn({p, _, _}) -> p end)
      |> Enum.zip(types)
      |> Map.new()

    {delta,true,fun_type}
  else
    #IO.puts "asdf"
    delta=para
      |> Enum.map(fn({p, _, _}) -> p end)
      |> Map.new(fn x -> {x,:none} end)
    {delta,false,:none}
  end
  #IO.inspect fun_type
  delta = Map.put(delta,:return,fun_type)

  inf_types = GPotion.TypeInference.infer_types(delta,body)

  fun_type = if is_typed do fun_type else Map.get(inf_types,:return) end

  param_list = para
    |> Enum.map(fn {p, _, _}-> gen_para(p,Map.get(inf_types,p)) end)
    |> Enum.join(", ")

  #types_para = para
   # |>  Enum.map(fn {p, _, _}-> Map.get(inf_types,p) end)

 #{param_list,_types_para,is_typed,inf_types,fun_type} = if is_list(List.last(para)) do
  #  [fun_type|types_para] = List.last(para)
  #  param_list = para
  #    |> List.delete_at(length(para)-1)
  #    |> Enum.map(fn({p, _, _}) -> p end)
  #    |> Enum.zip(types_para)
  #   |> Enum.map(fn({p,t}) -> gen_para(p,t) end)
  #   |> Enum.join(", ")
  # {param_list,types_para,true,%{},fun_type}
  #else
  #  types = para
  #  |> Enum.map(fn({p, _, _}) -> p end)
  #  |> Map.new(fn x -> {x,:none} end)
  #  |> Map.put(:return,:none)
  #  |> GPotion.TypeInference.infer_types(body)

  #  fun_type =  Map.get(types,:return)
    #IO.inspect fun_type
    #raise "hell"
   # param_list = para
   # |> Enum.map(fn {p, _, _}-> gen_para(p,Map.get(types,p)) end)
   # |> Enum.join(", ")

  #  types_para = para
   # |>  Enum.map(fn {p, _, _}-> Map.get(types,p) end)
   #{param_list,types_para,false,types,fun_type}

 #end
 cuda_body = GPotion.CudaBackend.gen_cuda(body,inf_types,is_typed)
 k = GPotion.CudaBackend.gen_function(fname,param_list,cuda_body,fun_type)
 #accessfunc = GPotion.CudaBackend.gen_kernel_call(fname,length(types_para),Enum.reverse(types_para))
 if(File.exists?("c_src/#{module_name}.cu")) do
  file = File.open!("c_src/#{module_name}.cu", [:append])
  IO.write(file, "\n" <> k <> "\n\n")
else
  file = File.open!("c_src/#{module_name}.cu", [:write])
  IO.write(file, "#include \"erl_nif.h\"\n\n" <> k <> "\n\n")
  File.close(file)
end
 #IO.puts k
 #IO.puts accessfunc
 #para = if is_list(List.last(para)) do List.delete_at(para,length(para)-1) else para end
 para = para
  |> Enum.map(fn {p, b, c}-> {String.to_atom("_" <> to_string(p)),b,c} end)


 quote do
    def unquote({fname,comp_info, para})do
      raise "A gp function can only be called in kernels"
    end
  end
end

  def create_ref_nif(_matrex) do
    raise "NIF create_ref_nif/1 not implemented"
end
def new_pinned_nif(_list,_length) do
  raise "NIF new_pinned_nif/1 not implemented"
end
def new_gmatrex_pinned_nif(_array) do
  raise "NIF new_gmatrex_pinned_nif/1 not implemented"
end
def new_pinned(list) do
  size = length(list)
  {new_pinned_nif(list,size), {1,size}}
end
def new_gmatrex(%Matrex{data: matrix} = a) do
  ref=create_ref_nif(matrix)
  {ref, Matrex.size(a)}
end
def new_gmatrex({array,{l,c}}) do
  ref=new_gmatrex_pinned_nif(array)
  {ref, {l,c}}
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
