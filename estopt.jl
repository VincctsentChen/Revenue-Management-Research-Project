using Gurobi, JuMP
using DataFrames, GLM
using Plots
using Distributions
using Random
using LinearAlgebra
using DelimitedFiles

#Function for simulating data
function Data(n,m,a,b,c,r,s,form="flat",seed=1)
	Random.seed!(seed)
	#Drawing random decision data
	x = rand(n)
	#Drawing random outcome data
	y,y_min,y_max = Outcome(x,a,b,c,r,s,form)
	#Setting all possible values for decisions
    d = collect(0.1:1/m:0.9)
	return x,y,y_min,y_max,d
end

#Function for simulating outcome
function Outcome(x,a,b,c,r,s,form="flat")
	n = length(x)
    y = zeros(n)
    #Simulating the random outcome for each random decision
    for i = 1:n
        if form == "flat"
            y[i] = a + s*randn()[1]
        elseif form == "flat-double"
            y[i] = a + s*randn()[1]
            if 0.5 <= x[i] <= 0.75
                y[i] = r*a + s*randn()[1]
            end
        elseif form == "linear"
            y[i] = a + b*x[i] + s*randn()[1]
        elseif form == "linear-double"
            y[i] = a + b*x[i] + s*randn()[1]
            if 0.5 <= x[i]
                y[i] = r*a + a*(1-r)*x[i] + s*randn()[1]
            end
        elseif form == "quadratic"
            y[i] = a + b*x[i] + c*x[i]^2 + s*randn()[1]
        elseif form == "quadratic-double"
            y[i] = a + b*x[i] + c*x[i]^2 + s*randn()[1]
            if 0.5 <= x[i]
                y[i] = r*a + (3*a*(1-r)+2*b+c)*x[i] + (-2*a*(1-r)-2*b-c)*x[i]^2 + s*randn()[1]
            end
        elseif form == "newsvendor"
        	demand = a + s*randn()[1]
        	y[i] = -(c*x[i] + r*b*max(demand-x[i],0) + b*max(x[i]-demand,0))
        elseif form == "supplychain-normal"
        	y[i] = x[i] * (quantile(Normal(c,r),(a-x[i])/(a+b))) + s*randn()[1]
        elseif form == "supplychain-pareto"
        	y[i] = x[i] * (quantile(Pareto(c,r),(a-x[i])/(a+b))) + s*randn()[1]
        end		       
    end
    #Scaling the outcome to unit scale
    y_min = minimum(y)
    y_max = maximum(y)
    y = (y - ones(n)*y_min)./(ones(n)*(y_max-y_min))
	return y,y_min,y_max
end

#Function for computing result
function Result(y_min,y_max,x,a,b,c,r,form="flat")
	#Computing the expected outcome for given decision
    if form == "flat"
        y = a
    elseif form == "flat-double"
        y = a
        if 0.5 <= x <= 0.75
            y = r*a
        end
    elseif form == "linear"
        y = a + b*x
    elseif form == "linear-double"
        y = a + b*x
        if 0.5 <= x
            y[i] = r*a + a*(1-r)*x[i]
        end
    elseif form == "quadratic"
        y = a + b*x + c*x^2
    elseif form == "quadratic-double"
        y = a + b*x + c*x^2
        if 0.5 <= x
            y = r*a + (3*a*(1-r)+2*b+c)*x + (-2*a*(1-r)-2*b-c)*x^2
        end
    elseif form == "newsvendor"
    	demand = a
    	y = -(c*x + r*b*max(demand-x,0) + b*max(x-demand,0))
    elseif form == "supplychain-normal"
        y = x * (quantile(Normal(c,r),(a-x)/(a+b)))
    elseif form == "supplychain-pareto"
        y = x * (quantile(Pareto(c,r),(a-x)/(a+b)))
    end
    #Scaling the outcome to unit scale
    y = (y-y_min)/(y_max-y_min)
	return y
end

#Function for jointly estimating and optimizing
function EWO(x,y,d,kappa,lambda,M,showoutput=false)
    n = length(x)
    m = length(d)
    #Defining optimization model
	model = Model()
	set_optimizer(model, Gurobi.Optimizer)
	set_optimizer_attributes(model, "OutputFlag" => 0)
    #Defining variables
    @variable(model, a)
    @variable(model, b)
    @variable(model, u[1:n], Bin)
    @variable(model, v[1:n], Bin)
    @variable(model, w[1:n], Bin)
    @variable(model, z[1:m], Bin)
    @variable(model, s[1:m] >= 0)
    @variable(model, t[1:n] >= 0)
	#Defining objective
    @objective(model, Max, sum(s[j] for j=1:m) - lambda * sum(t[i] for i=1:n))
    #Defining constraints
    @constraints(model, begin
        decision, sum(z[j] for j=1:m) == 1
        revenue_passive[j=1:m], s[j] <= a+b*d[j]
        revenue_active[j=1:m], s[j] <= M*z[j]
        data[i=1:n], u[i] >= v[i] + w[i] - 1
        data_upper[i=1:n], x[i] - sum(d[j]*z[j] for j=1:m) >= kappa - M*v[i]
        data_lower[i=1:n], sum(d[j]*z[j] for j=1:m) - x[i] >= kappa - M*w[i]
        estimation_upper[i=1:n], t[i] >= y[i]-a-b*x[i]-M*(1-u[i])
        estimation_lower[i=1:n], t[i] >= -y[i]+a+b*x[i]-M*(1-u[i])
    end)
    #Optimizing model
    optimize!(model)
    obj_val = objective_value(model)
    time_val = solve_time(model)
    a_val = value.(a)
    b_val = value.(b)
    z_val = value.(z)
    return obj_val,time_val,a_val,b_val,z_val
end

#Function for separately estimating and optimizing
function ETO(x,y,d,form="linear",showoutput=false)
    n = length(x)
    m = length(d)
    #Estimating outcome model
    df = DataFrame(outcome=y,decision=x)
    if form == "linear"
		ols = lm(@formula(outcome~decision),df)
		a,b = coef(ols)
		c = 0
    elseif form == "quadratic"
		ols = lm(@formula(outcome~decision+decision^2),df)
		a,b,c = coef(ols)
	end
	#Defining optimization model
    model = Model()
	set_optimizer(model, Gurobi.Optimizer)
	set_optimizer_attributes(model, "OutputFlag" => 0)    
	#Defining variables
    @variable(model, z[1:m], Bin)
    #Defining objective
   	@objective(model, Max, sum((a+b*d[j]+c*d[j]^2)*z[j] for j=1:m))
    #Defining constraints
    @constraints(model, begin
        decision, sum(z[j] for j=1:m) == 1
    end)
    #Optimizing model
    optimize!(model)
    obj_val = objective_value(model)
    time_val = solve_time(model)
    a_val = a
    b_val = b
	c_val = c
    z_val = value.(z)
    return obj_val,time_val,a_val,b_val,c_val,z_val
end



#Selecting setup
#setup = ["flat-double",100,0,0,4,1,0.05,1]
#setup = ["linear-double",100,100,0,0.75,1,0.05,1]
#setup = ["quadratic-double",100,100,-100,0.5,1,0.05,0.1]
setup = ["newsvendor",0.5,4,1,0.75,0.1,0.05,0.1]
#setup = ["supplychain-normal",1,1,4,3,0.1,0.05,0.1]
#setup = ["supplychain-pareto",1,1.5,0.25,1,0.1,0.05,0.1]

#Setting parameters
form_data = setup[1]
alpha = setup[2]
beta = setup[3]
gamma = setup[4]
rho = setup[5]
sigma = setup[6]
kappa = setup[7]
lambda = setup[8]
n = 1000
m = 100
big_M = 1000
seed_numbers = 100

#Printing parameter settings
println("Parameter settings:")
println("form_data: ", form_data)
println("alpha: ", alpha)
println("beta: ", beta)
println("gamma: ", gamma)
println("rho: ", rho)
println("sigma: ", sigma)
println("kappa: ", kappa)
println("lambda: ", lambda)
println("n: ", n)
println("m: ", m)
println("seed_numbers: ", seed_numbers)

#Initializing results table
Results = Array{Union{Missing, String, Int64, Float64}}(missing, 3*seed_numbers, 11+6)

seed_number = 1
#Running models for specified number of seeds to generate results
for seed_number = 1:seed_numbers
	println()
	println("seed_number: ", seed_number)

	#Initializing parameter settings in results table
	Results[3*(seed_number-1)+1,1:11] = [form_data,alpha,beta,gamma,rho,sigma,kappa,lambda,n,m,seed_number]
	Results[3*(seed_number-1)+2,1:11] = [form_data,alpha,beta,gamma,rho,sigma,kappa,lambda,n,m,seed_number]
	Results[3*(seed_number-1)+3,1:11] = [form_data,alpha,beta,gamma,rho,sigma,kappa,lambda,n,m,seed_number]

	#Generating data
	x_data,y_data,y_min_data,y_max_data,d_data = Data(n,m,alpha,beta,gamma,rho,sigma,form_data,seed_number)

	#Solving Estimate-While-Optimize model
	solution_EWO = EWO(x_data,y_data,d_data,kappa,lambda,big_M)

	#Saving solution and plotting parameters for Estimate-While-Optimize model
	a_EWO = solution_EWO[3]
	b_EWO = solution_EWO[4]
	z_EWO = solution_EWO[5]
	x_EWO = dot(d_data,z_EWO)
	y_EWO = Result(y_min_data,y_max_data,x_EWO,alpha,beta,gamma,rho,form_data)
	x_EWO_range = collect(x_EWO - kappa:(2*kappa/100):x_EWO + kappa)
	y_EWO_range = a_EWO*ones(length(x_EWO_range)) + b_EWO*x_EWO_range

	#Writing Estimate-While-Optimize results to results table
	Results[3*(seed_number-1)+1,12:17] = ["EWO",a_EWO,b_EWO,0,x_EWO[1],y_EWO[1]]

	#Printing Estimate-While-Optimize results
	println("EWO")
	println("a: ", a_EWO)
	println("b: ", b_EWO)
	println("x: ", x_EWO)
	println("y: ", y_EWO)

	#Solving Estimate-Then-Optimize-Linear model
	solution_ETOL = ETO(x_data,y_data,d_data,"linear")

	#Saving solution and plotting parameters for Estimate-Then-Optimize-Linear model
	a_ETOL = solution_ETOL[3]
	b_ETOL = solution_ETOL[4]
	c_ETOL = solution_ETOL[5]
	z_ETOL = solution_ETOL[6]
	x_ETOL = dot(d_data,z_ETOL)
	y_ETOL = Result(y_min_data,y_max_data,x_ETOL,alpha,beta,gamma,rho,form_data)
	x_ETOL_range = collect(0:1/100:1)
	y_ETOL_range = a_ETOL*ones(length(x_ETOL_range)) + b_ETOL*x_ETOL_range + c_ETOL*x_ETOL_range.^2

	#Writing Estimate-Then-Optimize-Linear results to results table
	Results[3*(seed_number-1)+2,12:17] = ["ETOL",a_ETOL,b_ETOL,c_ETOL,x_ETOL[1],y_ETOL[1]]

	#Printing Estimate-Then-Optimize-Linear results
	println("ETOL")
	println("a: ", a_ETOL)
	println("b: ", b_ETOL)
	println("c: ", c_ETOL)
	println("x: ", x_ETOL)
	println("y: ", y_ETOL)

	#Solving Estimate-Then-Optimize-Quadratic model
	solution_ETOQ = ETO(x_data,y_data,d_data,"quadratic")

	#Saving solution and plotting parameters for Estimate-Then-Optimize-Quadratic model
	a_ETOQ = solution_ETOQ[3]
	b_ETOQ = solution_ETOQ[4]
	c_ETOQ = solution_ETOQ[5]
	z_ETOQ = solution_ETOQ[6]
	x_ETOQ = dot(d_data,z_ETOQ)
	y_ETOQ = Result(y_min_data,y_max_data,x_ETOQ,alpha,beta,gamma,rho,form_data)
	x_ETOQ_range = collect(0:1/100:1)
	y_ETOQ_range = a_ETOQ*ones(length(x_ETOQ_range)) + b_ETOQ*x_ETOQ_range + c_ETOQ*x_ETOQ_range.^2

	#Writing Estimate-Then-Optimize-Quadratic results to results table
	Results[3*(seed_number-1)+3,12:17] = ["ETOQ",a_ETOQ,b_ETOQ,c_ETOQ,x_ETOQ[1],y_ETOQ[1]]

	#Printing Estimate-Then-Optimize-Quadratic results
	println("ETOQ")
	println("a: ", a_ETOQ)
	println("b: ", b_ETOQ)
	println("c: ", c_ETOQ)
	println("x: ", x_ETOQ)
	println("y: ", y_ETOQ)
end

#Writing results table to text file
writedlm("Results.txt", Results)
	
#Plotting data
gr()
p = plot(x_data,y_data,seriestype=:scatter,title="Objective",xlabel="Decision",ylabel="Outcome",label="Data",legendfontsize=5,markersize=2,markercolor="grey")
p = plot!(x_EWO_range,y_EWO_range,label="EWO",linewidth=2,linecolor="red")
p = plot!(x_ETOL_range,y_ETOL_range,label="ETOL",linewidth=2,linecolor="blue")
p = plot!(x_ETOQ_range,y_ETOQ_range,label="ETOQ",linewidth=2,linecolor="green")
plot(p)