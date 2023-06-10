defmodule GPotion.CudaBackend do
  def gen_kernel(name,para,body) do
    "__global__\nvoid #{name}(#{para})\n{\n#{body}\n}"
  end
  def gen_cuda(body,types,is_typed) do
    pid = spawn_link(fn -> types_server([],types,is_typed) end)
    Process.register(pid, :types_server)
    code = gen_body(body)
    send(pid,{:kill})
    code
  end
  def gen_body(body) do

    case body do
      {:__block__, _, _code} ->
        gen_block body
      {:do, {:__block__,pos, code}} ->
        gen_block {:__block__, pos,code}
      {:do, exp} ->
        gen_command exp
      {_,_,_} ->
        gen_command body
    end

  end
  defp gen_block({:__block__, _, code}) do
    code
        |>Enum.map(&gen_command/1)
        |>Enum.join("\n")
  end
  defp gen_header_for(header) do
    case header do
      {:in, _,[{var,_,nil},{:range,_,[n]}]} ->
            "for( int #{var} = 0; #{var}<#{gen_exp n}; #{var}++)"
      {:in, _,[{var,_,nil},{:range,_,[argr1,argr2]}]} ->
            "for( int #{var} = #{gen_exp argr1}; #{var}<#{gen_exp argr2}; #{var}++)"
      {:in, _,[{var,_,nil},{:range,_,[argr1,argr2,step]}]} ->
            "for( int #{var} = #{gen_exp argr1}; #{var}<#{gen_exp argr2}; #{var}+=#{gen_exp step})"
    end
  end
  defp gen_command(code) do
  #  if check_atrib_last code do
   #    gen_atrib_last code
   # else
      case code do
        {:for,_,[param,[body]]} ->
          header = gen_header_for(param)
          body = gen_body(body)
          header <> "{\n" <> body <> "\n}\n"
        {:=, _, [arg, exp]} ->
          a = gen_exp arg
          e = gen_exp exp
          case arg do
            {{:., _, [Access, :get]}, _, [_,_]} ->
              "\t#{a} = #{e}\;"
            _ ->
              send(:types_server,{:check_var, a, self()})
              receive do
                {:is_typed} ->
                  "\t#{a} = #{e}\;"
                {:type,type}->
                   "\t#{type} #{a} = #{e}\;"
                {:alredy_declared} ->
                   "\t#{a} = #{e}\;"
              end
          end


        {:if, _, if_com} ->
            genIf(if_com)
        {:do_while, _, [[doblock]]} ->
            "do{\n" <> gen_body(doblock)
        {:do_while_test, _, [exp]} ->
          "\nwhile("<> (gen_exp exp) <>  ");"
        {:while, _, [bexp,[body]]} ->
          "while(" <> (gen_exp bexp) <> "){\n" <> (gen_body body) <> "\n}"
        # CRIAÇÃO DE NOVOS VETORES
        {{:., _, [Access, :get]}, _, [arg1,arg2]} ->
            name = gen_exp arg1
            index = gen_exp arg2
            "float #{name}[#{index}];"
        {:__shared__,_ , [{{:., _, [Access, :get]}, _, [arg1,arg2]}]} ->
          name = gen_exp arg1
          index = gen_exp arg2
          "__shared__ float #{name}[#{index}];"
        {:var, _ , [{var,_,[{:=, _, [{type,_,nil}, exp]}]}]} ->
          #IO.puts "aqui"
          gexp = gen_exp exp
          "#{to_string type} #{to_string var} = #{gexp};"
        {:var, _ , [{var,_,[{:=, _, [type, exp]}]}]} ->
            gexp = gen_exp exp
            "#{to_string type} #{to_string var} = #{gexp};"
        {:var, _ , [{var,_,[{type,_,_}]}]} ->
           "#{to_string type} #{to_string var};"
        {:var, _ , [{var,_,[type]}]} ->
           "#{to_string type} #{to_string var};"
        {fun, _, args} when is_list(args)->
          nargs=args
          |> Enum.map(&gen_exp/1)
          |> Enum.join(", ")
          "#{fun}(#{nargs})\;"
        {str,_ ,_ } ->
            "#{to_string str};"
        number when is_integer(number) or is_float(number) -> to_string(number)
        #string when is_string(string)) -> string #to_string(number)
      end
    end
    defp gen_exp(exp) do
      case exp do
         {{:., _, [Access, :get]}, _, [arg1,arg2]} ->
          name = gen_exp arg1
          index = gen_exp arg2
          "#{name}[#{index}]"
        {{:., _, [{struct, _, nil}, field]},_,[]} ->
          "#{to_string struct}.#{to_string(field)}"
        {{:., _, [{:__aliases__, _, [struct]}, field]}, _, []} ->
          "#{to_string struct}.#{to_string(field)}"
        {op, _, args} when op in [:+, :-, :/, :*, :<=, :<, :>, :>=, :&&, :||, :!,:!=] ->
          case args do
            [a1] ->
              "(#{to_string(op)} #{gen_exp a1})"
            [a1,a2] ->
              "(#{gen_exp a1} #{to_string(op)} #{gen_exp a2})"
            end
        {var, _, nil} when is_atom(var) -> to_string(var)
        {fun, _, args} ->
          nargs=args
          |> Enum.map(&gen_exp/1)
          |> Enum.join(", ")
          "#{fun}(#{nargs})"
        number when is_integer(number) or is_float(number) -> to_string(number)
        string when is_binary(string)  -> "\"#{string}\""
      end

    end
    defp genIf([bexp, [do: then]]) do
        gen_then([bexp, [do: then]])
    end
    defp genIf([bexp, [do: thenbranch, else: elsebranch]]) do
         gen_then([bexp, [do: thenbranch]])
         <>
         "else{\n" <>
         (gen_body elsebranch) <>
         "\n}\n"
    end
    defp gen_then([bexp, [do: then]]) do
      "if(#{gen_exp bexp})\n" <>
      "{\n" <>
      (gen_body then) <>
      "\n}\n"
    end

#######

def types_server(used,types, is_typed) do
   if (is_typed) do
    receive do
      {:check_var, _var, pid} ->
          send(pid,{:is_typed})
          types_server(used,types, is_typed)
      {:kill} ->
            :ok
   end
  else
   receive do
    {:check_var, var, pid} ->
      if (!Enum.member?(used,var)) do
        type = Map.get(types,String.to_atom(var))
        if(type == nil) do
          IO.inspect var
          IO.inspect types
          raise "Could not find type for variable #{var}. Please declare it using \"var #{var} type\""
        end
        send(pid,{:type,type})
        types_server([var|used],types,is_typed)
      else
        send(pid,{:alredy_declared})
        types_server(used,types,is_typed)
      end
    {:kill} ->
      :ok
    end

  end
end

  def gen_kernel_call(kname,nargs,types) do
    gen_header(kname) <> gen_args(nargs,types) <> gen_call(kname,nargs)
  end

  def gen_header(fname) do
  "extern \"C\" void #{fname}_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type)
  {

    ERL_NIF_TERM list;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail;
    float **array_res;

    const ERL_NIF_TERM *tuple_blocks;
    const ERL_NIF_TERM *tuple_threads;
    int arity;

    if (!enif_get_tuple(env, argv[1], &arity, &tuple_blocks)) {
      printf (\"spawn: blocks argument is not a tuple\");
    }

    if (!enif_get_tuple(env, argv[2], &arity, &tuple_threads)) {
      printf (\"spawn:threads argument is not a tuple\");
    }
    int b1,b2,b3,t1,t2,t3;

    enif_get_int(env,tuple_blocks[0],&b1);
    enif_get_int(env,tuple_blocks[1],&b2);
    enif_get_int(env,tuple_blocks[2],&b3);
    enif_get_int(env,tuple_threads[0],&t1);
    enif_get_int(env,tuple_threads[1],&t2);
    enif_get_int(env,tuple_threads[2],&t3);

    dim3 blocks(b1,b2,b3);
    dim3 threads(t1,t2,t3);

    list= argv[3];

"
  end
  def gen_call(kernelname,nargs) do
"   #{kernelname}<<<blocks, threads>>>" <> gen_call_args(nargs) <> ";
    cudaError_t error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)
     { char message[200];
       strcpy(message,\"Error kernel call: \");
       strcat(message, cudaGetErrorString(error_gpu));
       enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
     }
}
"
  end
  def gen_call_args(nargs) do
    "(" <> gen_call_args_(nargs-1) <>"arg#{nargs})"
  end
  def gen_call_args_(0) do
    ""
  end
  def gen_call_args_(n) do
    args = gen_call_args_(n-1)
    args <> "arg#{n},"
  end
  def gen_args(0,_l) do
    ""
  end
  def gen_args(n,[]) do
    args = gen_args(n-1,[])
    arg = gen_arg_matrix(n)
    args <> arg
  end
  def gen_args(n,[:matrex|t]) do
    args = gen_args(n-1,t)
    arg = gen_arg_matrix(n)
    args <> arg
  end
  def gen_args(n,[:int|t]) do
    args = gen_args(n-1,t)
    arg = gen_arg_int(n)
    args <> arg
  end
  def gen_args(n,[:float|t]) do
    args = gen_args(n-1,t)
    arg = gen_arg_float(n)
    args <> arg
  end
  def gen_args(n,[:double|t]) do
    args = gen_args(n-1,t)
    arg = gen_arg_double(n)
    args <> arg
  end
  def gen_arg_matrix(narg) do
"  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg#{narg} = *array_res;
  list = tail;

"
  end
  def gen_arg_int(narg) do
"  enif_get_list_cell(env,list,&head,&tail);
  int arg#{narg};
  enif_get_int(env, head, &arg#{narg});
  list = tail;

"
  end
  def gen_arg_float(narg) do
"  enif_get_list_cell(env,list,&head,&tail);
  double darg#{narg};
  float arg#{narg};
  enif_get_double(env, head, &darg#{narg});
  arg#{narg} = (float) darg#{narg};
  list = tail;

"
  end
  def gen_arg_double(narg) do
"  enif_get_list_cell(env,list,&head,&tail);
  double arg#{narg};
  enif_get_double(env, head, &darg#{narg});
  list = tail;

"
  end
end
